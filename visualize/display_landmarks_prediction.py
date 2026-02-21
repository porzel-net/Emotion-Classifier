"""Show predictions from the trained landmark detector on the test split."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import tensorflow as tf

from helpers.landmark_data import DEFAULT_SIZE, load_landmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display landmark detector predictions.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/cropped-face-keypoint-dataset-68-landmarks"),
        help="Root folder containing the annotation CSV and image directories.",
    )
    parser.add_argument("--csv", type=Path, default=Path("test.csv"), help="CSV file with test annotations.")
    parser.add_argument("--images", type=Path, default=Path("test"), help="Subdirectory with test images.")
    parser.add_argument("--model", type=Path, default=Path("models/landmarks_detector.keras"), help="Trained model to load.")
    parser.add_argument("--rows", type=int, default=3, help="Number of rows in the preview grid.")
    parser.add_argument("--cols", type=int, default=3, help="Number of columns in the preview grid.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size used when running the model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.data_dir / args.csv
    image_root = args.data_dir / args.images

    print(f"Loading test data from {csv_path}")
    images, ground_truth = load_landmarks(csv_path, image_root, target_size=DEFAULT_SIZE)
    stats_path = args.model.parent / "landmarks_stats.npz"
    if stats_path.exists():
        with np.load(stats_path) as stats:
            mean = stats["mean"]
            std = stats["std"]
        standardized_images = (images - mean) / std
    else:
        standardized_images = images

    print(f"Loading model from {args.model}")
    model = tf.keras.models.load_model(args.model)
    predictions = np.asarray(model.predict(standardized_images, batch_size=args.batch_size), dtype=np.float32)

    rows = args.rows
    cols = args.cols
    page_size = rows * cols
    total_pages = max(1, math.ceil(len(images) / page_size))
    page_index = 0

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    plt.subplots_adjust(top=0.9, bottom=0.08, hspace=0.3, wspace=0.3)

    legend_handles = [
        Line2D([], [], color="lime", marker="o", linestyle="None", markersize=5, label="ground truth"),
        Line2D([], [], color="red", marker="o", linestyle="None", markersize=5, label="prediction"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.95, 0.95))

    status_text = fig.text(
        0.5,
        0.03,
        "Use ← / → to flip pages. Green = ground truth, Red = prediction.",
        ha="center",
        fontsize="small",
    )

    def draw_page() -> None:
        start = page_index * page_size
        fig.suptitle(f"Landmark predictions (page {page_index + 1}/{total_pages})", fontsize="large")
        for ax_idx, ax in enumerate(axes):
            sample_idx = start + ax_idx
            ax.clear()
            if sample_idx >= len(images):
                ax.axis("off")
                continue
            ax.imshow(images[sample_idx])
            try:
                gt_landmarks = ground_truth[sample_idx].reshape(-1, 2)
                pred_landmarks = predictions[sample_idx].reshape(-1, 2)
            except ValueError:
                ax.set_title("Unexpected landmark shape", color="red")
                continue
            ax.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], color="lime", s=10)
            ax.scatter(pred_landmarks[:, 0], pred_landmarks[:, 1], color="red", s=8, alpha=0.7)
            ax.set_title(f"Sample {sample_idx + 1}", fontsize="small")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.canvas.draw_idle()

    def on_key(event: plt.KeyEvent) -> None:
        nonlocal page_index
        if event.key == "right":
            page_index = min(total_pages - 1, page_index + 1)
            draw_page()
        elif event.key == "left":
            page_index = max(0, page_index - 1)
            draw_page()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw_page()
    plt.show()


if __name__ == "__main__":
    main()
