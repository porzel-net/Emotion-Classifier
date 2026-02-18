"""Visualize solo-face crop predictions side-by-side with ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.patches import Rectangle
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display SoloFace crop predictions.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/soloface-detection-dataset"),
        help="Root folder containing the train/val/test splits.",
    )
    parser.add_argument("--subset", type=str, default="test", help="Which split to visualize.")
    parser.add_argument("--model", type=Path, default=Path("models/solo_cropper.keras"), help="Trained model path.")
    parser.add_argument("--image-size", type=int, default=64, help="Input size used during training.")
    parser.add_argument("--rows", type=int, default=3, help="Number of rows in the grid.")
    parser.add_argument("--cols", type=int, default=3, help="Number of columns in the grid.")
    parser.add_argument("--start", type=int, default=0, help="Start index in the subset.")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to show.")
    return parser.parse_args()


def load_subset(
    data_root: Path,
    subset: str,
    max_samples: int | None = None,
) -> list[tuple[Path, int, list[float]]]:
    subset_dir = data_root / subset
    images_dir = subset_dir / "images"
    labels_dir = subset_dir / "labels"
    samples: list[tuple[Path, int, list[float]]] = []

    for label_path in sorted(labels_dir.glob("*.json")):
        if max_samples is not None and len(samples) >= max_samples:
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
        samples.append((image_path, int(record.get("class", 1)), [float(coord) for coord in record.get("bbox", [0.0, 0.0, 0.01, 0.01])]))

    if not samples:
        raise ValueError(f"No samples found under {subset_dir}.")
    return samples


def preprocess_images(
    samples: list[tuple[Path, int, list[float]]], image_size: tuple[int, int]
) -> tuple[np.ndarray, list[np.ndarray]]:
    inputs: list[np.ndarray] = []
    originals: list[np.ndarray] = []
    for image_path, _, _ in samples:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            originals.append(np.asarray(img))
            resized = img.resize(image_size, Image.BILINEAR)
            inputs.append(np.asarray(resized, dtype=np.float32) / 255.0)
    return np.stack(inputs), originals


def draw_bbox(ax, bbox, width, height, color, label):
    xmin, ymin, xmax, ymax = np.clip(bbox, 0.0, 1.0)
    rect = Rectangle(
        (xmin * width, ymin * height),
        (xmax - xmin) * width,
        (ymax - ymin) * height,
        linewidth=2,
        edgecolor=color,
        facecolor="none",
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            xmin * width + 2,
            ymin * height + 2,
            label,
            color=color,
            fontsize="small",
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )


def main() -> None:
    args = parse_args()
    image_size = (args.image_size, args.image_size)
    subset_samples = load_subset(args.data_dir, args.subset, max_samples=args.max_samples)
    if not subset_samples:
        raise ValueError("No samples found for display.")

    inputs, originals = preprocess_images(subset_samples, image_size)

    model = tf.keras.models.load_model(args.model)
    predictions = model.predict(inputs, verbose=0)
    pred_bboxes = predictions["bbox"]
    pred_confidence = predictions["confidence"].flatten()

    grid_size = args.rows * args.cols
    total_pages = max(1, (len(subset_samples) + grid_size - 1) // grid_size)
    start_page = min(max(0, args.start // grid_size), total_pages - 1)
    page_index = start_page

    fig, axes = plt.subplots(args.rows, args.cols, figsize=(args.cols * 3, args.rows * 3))
    axes_list = axes.flatten()
    plt.subplots_adjust(top=0.88, hspace=0.3, wspace=0.3)
    status_text = fig.text(
        0.5,
        0.03,
        f"Page {page_index + 1}/{total_pages}. Use ← / → to flip pages.",
        ha="center",
        fontsize="small",
    )

    def draw_page() -> None:
        start_idx = page_index * grid_size
        end_idx = min(start_idx + grid_size, len(subset_samples))
        fig.suptitle(
            f"SoloFace predictions ({args.subset}) [{start_idx}:{end_idx}]",
            fontsize="medium",
        )
        for ax_idx, ax in enumerate(axes_list):
            sample_idx = start_idx + ax_idx
            ax.clear()
            ax.axis("off")
            if sample_idx >= len(subset_samples):
                continue
            image_arr = originals[sample_idx]
            _, class_id, gt_bbox = subset_samples[sample_idx]
            height, width = image_arr.shape[:2]
            ax.imshow(image_arr)
            draw_bbox(
                ax,
                gt_bbox,
                width,
                height,
                "lime",
                f"GT face: {bool(class_id)}",
            )
            draw_bbox(
                ax,
                pred_bboxes[sample_idx],
                width,
                height,
                "red",
                f"pred conf {pred_confidence[sample_idx]:.2f}",
            )
        status_text.set_text(
            f"Page {page_index + 1}/{total_pages}. Use ← / → to flip pages."
        )
        fig.canvas.draw_idle()

    def on_key(event: plt.KeyEvent) -> None:
        nonlocal page_index
        if event.key == "right":
            if page_index < total_pages - 1:
                page_index += 1
                draw_page()
        elif event.key == "left":
            if page_index > 0:
                page_index -= 1
                draw_page()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw_page()
    handles = [
        Rectangle((0, 0), 1, 1, edgecolor="lime", facecolor="none", linewidth=2),
        Rectangle((0, 0), 1, 1, edgecolor="red", facecolor="none", linewidth=2),
    ]
    fig.legend(handles, ["ground truth bbox", "predicted bbox"], loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
