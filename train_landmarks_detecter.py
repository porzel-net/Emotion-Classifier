"""Train a compact landmark detector using the cropped-face-keypoint dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import apply_affine_transform

from landmark_data import DEFAULT_SIZE, load_landmarks


class LearningRatePrinter(tf.keras.callbacks.Callback):
    """Log the optimizer learning rate at epoch end."""

    def _resolve_learning_rate(self) -> float:
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)
        return float(tf.keras.backend.get_value(lr))

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        lr = self._resolve_learning_rate()
        print(f"Epoch {epoch + 1}: learning rate is {lr:.6g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 64×64 landmark detector.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/cropped-face-keypoint-dataset-68-landmarks"),
        help="Root folder containing the CSV and image subdirectories.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("training.csv"),
        help="CSV file (relative to data-dir) with the landmark annotations.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("training"),
        help="Subdirectory (relative to data-dir) containing the face crops.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/landmarks_detector.keras"),
        help="Where to write the best model checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction reserved for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of samples used (useful for quick experiments).",
    )
    parser.add_argument(
        "--augmentation-rounds",
        type=int,
        default=1,
        help="Number of augmented copies to add per sample (rounds ≥ 0).",
    )
    parser.add_argument(
        "--max-rotation",
        type=float,
        default=10.0,
        help="Maximum rotation in degrees for augmentation.",
    )
    parser.add_argument(
        "--min-scale",
        type=float,
        default=0.85,
        help="Minimum zoom scale to shrink the face (must be <= 1.0).",
    )
    parser.add_argument(
        "--max-translation",
        type=float,
        default=0.1,
        help="Maximum translation relative to width/height for augmentation.",
    )
    parser.add_argument(
        "--noise-prob",
        type=float,
        default=0.5,
        help="Probability of adding gaussian noise to each augmentation (0 to disable).",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.02,
        help="Standard deviation of gaussian noise added during augmentation.",
    )
    parser.add_argument(
        "--standardize",
        dest="standardize",
        action="store_true",
        help="Enable pixel standardization before training (default).",
    )
    parser.add_argument(
        "--no-standardize",
        dest="standardize",
        action="store_false",
        help="Skip pixel standardization and train on raw [0,1] inputs.",
    )
    parser.set_defaults(standardize=True)
    parser.add_argument(
        "--reduce-lr",
        choices=("plateau", "none"),
        default="plateau",
        help="Control the learning rate scheduler applied during training.",
    )
    parser.add_argument(
        "--reduce-lr-factor",
        type=float,
        default=0.5,
        help="Multiplicative factor for ReduceLROnPlateau (if used).",
    )
    parser.add_argument(
        "--reduce-lr-patience",
        type=int,
        default=3,
        help="Number of epochs with no improvement before reducing LR.",
    )
    parser.add_argument(
        "--reduce-lr-min-lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for ReduceLROnPlateau (if used).",
    )
    return parser.parse_args()


def split_data(images: np.ndarray, landmarks: np.ndarray, val_split: float, seed: int):
    idx = np.arange(len(images))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    val_count = max(1, int(len(idx) * val_split)) if val_split > 0 else 0
    val_count = min(val_count, len(idx) - 1)
    val_idx = idx[:val_count]
    train_idx = idx[val_count:]
    return (
        images[train_idx],
        landmarks[train_idx],
        images[val_idx],
        landmarks[val_idx],
    )


def _transform_landmarks(
    landmarks: np.ndarray,
    angle: float,
    scale: float,
    center: np.ndarray,
    translation: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the same rotation and scaling to the landmark coordinates."""
    rad = np.deg2rad(angle)
    cos = np.cos(rad)
    sin = np.sin(rad)
    matrix = scale * np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
    coords = landmarks.reshape(-1, 2)
    transformed = (coords - center) @ matrix.T + center
    if translation is not None:
        transformed += translation
    return transformed.reshape(-1)


def augment_dataset(
    images: np.ndarray,
    landmarks: np.ndarray,
    rounds: int,
    max_rotation: float,
    min_scale: float,
    max_translation: float,
    noise_prob: float,
    noise_scale: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Add slightly rotated and smaller versions of the dataset for diversity."""
    if rounds <= 0:
        return images, landmarks

    rng = np.random.default_rng(seed)
    height, width = images.shape[1], images.shape[2]
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)

    batches_images = [images]
    batches_landmarks = [landmarks]
    for _ in range(rounds):
        augmented_images = []
        augmented_landmarks = []
        for image, landmark in zip(images, landmarks):
            angle = float(rng.uniform(-max_rotation, max_rotation))
            scale = float(rng.uniform(min_scale, 1.0))
            shift_x = float(rng.uniform(-max_translation, max_translation)) * width
            shift_y = float(rng.uniform(-max_translation, max_translation)) * height
            augmented_image = apply_affine_transform(
                image,
                theta=angle,
                zx=scale,
                zy=scale,
                tx=shift_x,
                ty=shift_y,
                row_axis=0,
                col_axis=1,
                channel_axis=2,
                fill_mode="reflect",
                cval=0.0,
            )
            if noise_prob > 0 and rng.uniform() < noise_prob:
                noise = rng.normal(loc=0.0, scale=noise_scale, size=image.shape)
                augmented_image = np.clip(augmented_image + noise, 0.0, 1.0)
            augmented_images.append(augmented_image)
            translation = np.array([shift_x, shift_y], dtype=np.float32)
            augmented_landmarks.append(
                _transform_landmarks(landmark, angle, scale, center, translation)
            )
        batches_images.append(np.stack(augmented_images))
        batches_landmarks.append(np.stack(augmented_landmarks))

    return (
        np.concatenate(batches_images, axis=0),
        np.concatenate(batches_landmarks, axis=0),
    )


def standardize_images(
    images: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None:
        mean = images.mean(axis=(0, 1, 2), keepdims=True)
    if std is None:
        std = images.std(axis=(0, 1, 2), keepdims=True)
    std = np.maximum(std, 1e-6)
    standardized = (images - mean) / std
    return standardized, mean, std


def build_conv_block(x: tf.Tensor, filters: int, dropout_rate: float) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


def build_model(input_shape: tuple[int, int, int], output_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = build_conv_block(inputs, 12, dropout_rate=0.0)
    x = tf.keras.layers.Conv2D(24, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(output_dim, name="landmarks")(x)
    return tf.keras.Model(inputs, outputs, name="landmark_detector")


def make_dataset(images: np.ndarray, landmarks: np.ndarray, batch_size: int) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((images, landmarks))
    dataset = dataset.shuffle(buffer_size=len(images))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    args = parse_args()
    csv_path = args.data_dir / args.csv
    image_root = args.data_dir / args.images

    tf.get_logger().info("Loading dataset from %s", csv_path)
    images, landmarks = load_landmarks(
        csv_path,
        image_root,
        target_size=DEFAULT_SIZE,
        max_samples=args.max_samples,
    )
    tf.get_logger().info("Loaded %d samples.", len(images))

    (
        train_images,
        train_landmarks,
        val_images,
        val_landmarks,
    ) = split_data(images, landmarks, args.val_split, args.seed)
    train_images, train_landmarks = augment_dataset(
        train_images,
        train_landmarks,
        rounds=args.augmentation_rounds,
        max_rotation=args.max_rotation,
        min_scale=args.min_scale,
        max_translation=args.max_translation,
        noise_prob=args.noise_prob,
        noise_scale=args.noise_scale,
        seed=args.seed,
    )
    if args.standardize:
        train_images, mean, std = standardize_images(train_images)
        val_images, _, _ = standardize_images(val_images, mean, std)
    else:
        channel_dim = train_images.shape[-1]
        mean = np.zeros((1, 1, channel_dim), dtype=np.float32)
        std = np.ones((1, 1, channel_dim), dtype=np.float32)

    stats_path = args.output.parent / "landmarks_stats.npz"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(stats_path, mean=mean, std=std)

    train_ds = make_dataset(train_images, train_landmarks, args.batch_size)
    val_ds = make_dataset(val_images, val_landmarks, args.batch_size)

    model = build_model((*DEFAULT_SIZE, 3), output_dim=train_landmarks.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    print("Model summary:")
    model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        args.output,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    callbacks: list[tf.keras.callbacks.Callback] = [checkpoint]
    if args.reduce_lr == "plateau":
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=args.reduce_lr_factor,
                patience=args.reduce_lr_patience,
                min_lr=args.reduce_lr_min_lr,
                verbose=1,
            )
        )
    lr_logger = LearningRatePrinter()
    callbacks.append(lr_logger)

    model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )


if __name__ == "__main__":
    main()
