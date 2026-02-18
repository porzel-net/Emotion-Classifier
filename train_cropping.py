from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a face cropping model on SoloFace.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/soloface-detection-dataset"),
        help="Root folder containing the train/val/test splits.",
    )
    parser.add_argument("--train-split", type=str, default="train", help="Subdirectory to use for training.")
    parser.add_argument("--val-split", type=str, default="val", help="Subdirectory to use for validation.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/solo_cropper.keras"),
        help="Path where the best model is saved.",
    )
    parser.add_argument("--image-size", type=int, default=64, help="Square size for model inputs.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit training samples.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Limit validation samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--augment", action="store_true", help="Enable horizontal flip + color jitter augmentation.")
    parser.add_argument(
        "--augment-translation",
        type=float,
        default=10.0,
        help="Max translation fraction (0-100) applied during augmentation.",
    )
    parser.add_argument(
        "--augment-zoom",
        type=float,
        default=0.1,
        help="Max zoom factor applied during augmentation (e.g., 0.1 => +-10%).",
    )
    flip_group = parser.add_mutually_exclusive_group()
    flip_group.add_argument(
        "--augment-flip",
        dest="augment_flip",
        action="store_true",
        help="Enable random horizontal flip when augmentation is active.",
    )
    flip_group.add_argument(
        "--no-augment-flip",
        dest="augment_flip",
        action="store_false",
        help="Disable horizontal flip even if augmentation is enabled.",
    )
    color_group = parser.add_mutually_exclusive_group()
    color_group.add_argument(
        "--augment-color-jitter",
        dest="augment_color_jitter",
        action="store_true",
        help="Enable random brightness/contrast jitter during augmentation.",
    )
    color_group.add_argument(
        "--no-augment-color-jitter",
        dest="augment_color_jitter",
        action="store_false",
        help="Disable color jitter even when augmentation is active.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=("lite", "mobilenet_v3_small"),
        default="lite",
        help="Select base feature extractor for regression.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze backbone weights for initial training.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate applied before the heads.",
    )
    parser.add_argument(
        "--confidence-loss-weight",
        type=float,
        default=0.8,
        help="Weight applied to the confidence loss.",
    )
    parser.add_argument(
        "--bbox-loss-weight",
        type=float,
        default=1.2,
        help="Weight applied to the bbox regression loss.",
    )
    parser.add_argument(
        "--bbox-loss-delta",
        type=float,
        default=0.1,
        help="Delta parameter for the Huber bbox loss (higher -> more linear).",
    )
    parser.add_argument(
        "--boundary-penalty",
        type=float,
        default=0.0,
        help="Additional multiplier applied to bbox loss for outlying crops.",
    )
    parser.add_argument(
        "--boundary-penalty-threshold",
        type=float,
        default=0.0,
        help="Normalized distance to the border under which the boundary penalty scales the bbox loss (0-0.5).",
    )
    parser.add_argument(
        "--undersample-boundary-threshold",
        type=float,
        default=0.0,
        help="Normalized distance to the image border below which samples may be dropped (0-0.5).",
    )
    parser.add_argument(
        "--undersample-boundary-rate",
        type=float,
        default=0.0,
        help="Fraction of samples near the border to drop during training (0-1).",
    )
    parser.add_argument(
        "--disable-reduce-lr",
        action="store_true",
        help="Skip the ReduceLROnPlateau callback when training.",
    )
    parser.set_defaults(augment_flip=True, augment_color_jitter=True)
    return parser.parse_args()


def load_split(
    data_root: Path,
    subset: str,
    image_size: tuple[int, int],
    max_samples: int | None = None,
) -> tuple[Sequence[str], np.ndarray, np.ndarray]:
    """Return image paths, class ids, and normalized bounding boxes."""
    subset_dir = data_root / subset
    images_dir = subset_dir / "images"
    labels_dir = subset_dir / "labels"
    image_paths: list[str] = []
    classes: list[float] = []
    bboxes: list[list[float]] = []

    for label_path in sorted(labels_dir.glob("*.json")):
        if max_samples is not None and len(image_paths) >= max_samples:
            break
        with label_path.open("r") as handle:
            record = json.load(handle)
        image_name = record.get("image")
        image_path = images_dir / image_name
        if not image_path.exists():
            fallback = images_dir / f"{label_path.stem}.jpg"
            if fallback.exists():
                image_path = fallback
            else:
                continue
        image_paths.append(str(image_path))
        classes.append(float(record.get("class", 1)))
        bboxes.append([float(coord) for coord in record.get("bbox", [0.0, 0.0, 0.01, 0.01])])

    if not image_paths:
        raise ValueError(f"No samples found in {subset_dir}.")

    return image_paths, np.array(classes, dtype=np.float32), np.array(bboxes, dtype=np.float32)


def _load_sample(path: str, class_id: tf.Tensor, bbox: tf.Tensor, image_size: tuple[int, int]):
    image_data = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, image_size, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32) / 255.0
    label = {
        "confidence": tf.reshape(tf.cast(class_id, tf.float32), (1,)),
        "bbox": tf.clip_by_value(bbox, 0.0, 1.0),
    }
    return image, label


def _jitter_bbox(bbox: tf.Tensor, translate_frac: float, zoom_frac: float) -> tf.Tensor:
    bbox = tf.reshape(bbox, (4,))
    xmin, ymin, xmax, ymax = tf.split(bbox, 4)
    width = xmax - xmin
    height = ymax - ymin
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    if translate_frac > 0:
        shift_x = tf.random.uniform([], -translate_frac, translate_frac)
        shift_y = tf.random.uniform([], -translate_frac, translate_frac)
        center_x = tf.clip_by_value(center_x + shift_x, width / 2, 1 - width / 2)
        center_y = tf.clip_by_value(center_y + shift_y, height / 2, 1 - height / 2)

    if zoom_frac > 0:
        scale = 1 + tf.random.uniform([], -zoom_frac, zoom_frac)
        width = tf.clip_by_value(width * scale, 1e-3, 1.0)
        height = tf.clip_by_value(height * scale, 1e-3, 1.0)

    xmin = tf.clip_by_value(center_x - width / 2, 0.0, 1.0)
    ymin = tf.clip_by_value(center_y - height / 2, 0.0, 1.0)
    xmax = tf.clip_by_value(center_x + width / 2, 0.0, 1.0)
    ymax = tf.clip_by_value(center_y + height / 2, 0.0, 1.0)
    return tf.concat([xmin, ymin, xmax, ymax], axis=0)


def _augment_sample(
    image: tf.Tensor,
    label: dict[str, tf.Tensor],
    translate_frac: float,
    zoom_frac: float,
    flip: bool,
    color_jitter: bool,
):
    bbox = label["bbox"]
    if flip and tf.random.uniform([]) < 0.5:
        image = tf.image.flip_left_right(image)
        xmin, ymin, xmax, ymax = tf.split(bbox, 4, axis=-1)
        bbox = tf.concat([1.0 - xmax, ymin, 1.0 - xmin, ymax], axis=-1)
    bbox = _jitter_bbox(bbox, translate_frac, zoom_frac)
    if color_jitter:
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
    bbox = tf.clip_by_value(bbox, 0.0, 1.0)
    return image, {"confidence": label["confidence"], "bbox": bbox}


def build_dataset(
    image_paths: Sequence[str],
    classes: np.ndarray,
    bboxes: np.ndarray,
    image_size: tuple[int, int],
    batch_size: int,
    shuffle: bool = True,
    augment: bool = False,
    augment_translation: float = 0.0,
    augment_zoom: float = 0.0,
    augment_flip: bool = True,
    augment_color_jitter: bool = True,
    undersample_boundary_threshold: float = 0.0,
    undersample_boundary_rate: float = 0.0,
    seed: int | None = None,
):
    dataset = tf.data.Dataset.from_tensor_slices((list(image_paths), classes, bboxes))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=seed)
    if undersample_boundary_threshold > 0.0 or undersample_boundary_rate > 0.0:
        threshold = tf.constant(undersample_boundary_threshold, dtype=tf.float32)
        rate = tf.constant(max(min(undersample_boundary_rate, 1.0), 0.0), dtype=tf.float32)

        def _keep_boundary_sample(path, class_id, bbox):  # type: ignore[no-untyped-def]
            dist = tf.reduce_min(tf.stack([bbox[0], bbox[1], 1.0 - bbox[2], 1.0 - bbox[3]], axis=0))
            near_border = tf.less(dist, threshold)
            drop_prob = tf.random.uniform([], dtype=tf.float32)
            keep = tf.logical_or(tf.logical_not(near_border), tf.greater_equal(drop_prob, rate))
            return keep

        dataset = dataset.filter(_keep_boundary_sample)
    dataset = dataset.map(
        lambda path, class_id, bbox: _load_sample(path, class_id, bbox, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if augment:
        dataset = dataset.map(
            lambda image, label: _augment_sample(
                image,
                label,
                augment_translation,
                augment_zoom,
                augment_flip,
                augment_color_jitter,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _lite_conv_block(x: tf.Tensor, filters: int, stride: int = 1) -> tf.Tensor:
    shortcut = layers.Conv2D(filters, 1, padding="same", strides=stride, use_bias=False)(x)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.SeparableConv2D(filters, 3, padding="same", strides=stride, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("swish")(x)
    return x


def _build_lite_backbone(inputs: tf.Tensor) -> tf.Tensor:
    x = inputs
    block_params = (
        (16, 1),
        (24, 2),
        (40, 2),
    )
    for filters, stride in block_params:
        x = _lite_conv_block(x, filters, stride=stride)
    for filters in (80,):
        x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("swish")(x)
    return x


def build_boundary_aware_bbox_loss(
    delta: float,
    penalty_weight: float,
    penalty_threshold: float,
) -> tf.keras.losses.Loss | Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    delta = max(delta, 0.0)
    penalty_weight = max(penalty_weight, 0.0)
    penalty_threshold = max(min(penalty_threshold, 0.5), 0.0)

    base_loss = tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.NONE)

    def _boundary_dist(bbox: tf.Tensor) -> tf.Tensor:
        bbox = tf.clip_by_value(bbox, 0.0, 1.0)
        mins = tf.stack(
            [bbox[..., 0], bbox[..., 1], 1.0 - bbox[..., 2], 1.0 - bbox[..., 3]],
            axis=-1,
        )
        return tf.reduce_min(mins, axis=-1)

    def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        per_sample = base_loss(y_true, y_pred)
        if penalty_weight <= 0.0 or penalty_threshold <= 0.0:
            return tf.reduce_mean(per_sample)
        dist = _boundary_dist(y_pred)
        margin = tf.clip_by_value((penalty_threshold - dist) / penalty_threshold, 0.0, 1.0)
        penalty = margin * penalty_weight * tf.reduce_mean(tf.abs(y_true - y_pred), axis=-1)
        return tf.reduce_mean(per_sample + penalty)

    return _loss


def build_model(
    input_shape: tuple[int, int, int],
    backbone: Literal["lite", "mobilenet_v3_small"] = "lite",
    dropout_rate: float = 0.2,
    freeze_backbone: bool = False,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    dropout_rate = min(max(dropout_rate, 0.0), 0.5)

    if backbone == "mobilenet_v3_small":
        x = layers.Rescaling(2.0, offset=-1.0)(inputs)
        base = tf.keras.applications.MobileNetV3Small(
            include_top=False,
            input_shape=input_shape,
            weights=None,
        )
        base.trainable = not freeze_backbone
        x = base(x)
    else:
        x = _build_lite_backbone(inputs)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="swish")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(48, activation="swish")(x)

    confidence = layers.Dense(1, activation="sigmoid", name="confidence")(x)
    bbox = layers.Dense(4, activation="sigmoid", name="bbox")(x)
    return tf.keras.Model(inputs, {"confidence": confidence, "bbox": bbox}, name="solo_cropper")


def main() -> None:
    args = parse_args()
    tf.random.set_seed(args.seed)

    image_size = (args.image_size, args.image_size)
    train_paths, train_classes, train_bboxes = load_split(
        args.data_dir,
        args.train_split,
        image_size,
        max_samples=args.max_train_samples,
    )
    val_paths, val_classes, val_bboxes = load_split(
        args.data_dir,
        args.val_split,
        image_size,
        max_samples=args.max_val_samples,
    )

    print(f"Training samples: {len(train_paths)}; Validation samples: {len(val_paths)}")

    translate_frac = max(min(args.augment_translation / 100.0, 0.5), 0.0)
    zoom_frac = max(min(args.augment_zoom, 0.5), 0.0)
    train_ds = build_dataset(
        train_paths,
        train_classes,
        train_bboxes,
        image_size,
        batch_size=args.batch_size,
        shuffle=True,
        augment=args.augment,
        augment_translation=translate_frac,
        augment_zoom=zoom_frac,
        augment_flip=args.augment_flip,
        augment_color_jitter=args.augment_color_jitter,
        undersample_boundary_threshold=args.undersample_boundary_threshold,
        undersample_boundary_rate=args.undersample_boundary_rate,
        seed=args.seed,
    )
    val_ds = build_dataset(
        val_paths,
        val_classes,
        val_bboxes,
        image_size,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
    )

    model = build_model(
        (*image_size, 3),
        backbone=args.backbone,
        dropout_rate=args.dropout,
        freeze_backbone=args.freeze_backbone,
    )
    bbox_loss = build_boundary_aware_bbox_loss(
        args.bbox_loss_delta,
        args.boundary_penalty,
        args.boundary_penalty_threshold,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss={
            "confidence": tf.keras.losses.BinaryCrossentropy(),
            "bbox": bbox_loss,
        },
        loss_weights={
            "confidence": max(args.confidence_loss_weight, 0.1),
            "bbox": max(args.bbox_loss_weight, 0.1),
        },
        metrics={
            "confidence": tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            "bbox": tf.keras.metrics.MeanAbsoluteError(name="mae"),
        },
    )
    print("Model summary:")
    model.summary()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        args.output,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    staged_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        args.output.parent
        / f"{args.output.stem}-epoch{{epoch:02d}}-val{{val_loss:.4f}}{args.output.suffix}",
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
        mode="min",
    )
    callbacks = [checkpoint, staged_checkpoint]
    if not args.disable_reduce_lr:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                patience=4,
                factor=0.5,
                min_lr=1e-6,
                verbose=1,
            )
        )

    model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )


if __name__ == "__main__":
    main()
