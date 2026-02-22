#!/usr/bin/env python3
"""Evaluate an emotion dataset split and generate a confusion matrix."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from helpers.neconet_helpers import EMOTION_LABELS, get_device
from scripts.score_folder import (
    CLASS_ORDER,
    IMAGE_EXTENSIONS,
    build_eval_transform,
    build_evaluation_model,
    prepare_face_sample,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a labeled split folder and create confusion matrix outputs."
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to split folder containing class subfolders (recursive scan).",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=Path("confusion_matrix.png"),
        help="PNG path for confusion matrix plot.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("confusion_matrix.csv"),
        help="CSV path for confusion matrix values.",
    )
    parser.add_argument(
        "--weights",
        "--weights-dir",
        dest="weights",
        type=Path,
        default=Path("models/emotion-classifier-best.pth"),
        help="Checkpoint path.",
    )
    parser.add_argument("--device", default=None, help="Torch device to use (cpu/mps/cuda).")
    parser.add_argument(
        "--filter",
        choices=["sobel", "roberts", "laplacian", "none"],
        default="sobel",
        help="Input preprocessing filter.",
    )
    parser.add_argument("--width-multiplier", type=float, default=None, help="Optional width override.")
    parser.add_argument("--use-landmarks", action="store_true", help="Compatibility flag.")
    parser.add_argument("--use-gnn", action="store_true", help="Compatibility flag.")
    parser.add_argument("--gnn-hidden-dim", type=int, default=None, help="Optional GNN hidden dim override.")
    parser.add_argument("--gnn-steps", type=int, default=2, help="GNN message steps.")
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
        help="Compatibility flag; not used (landmarks are computed online when face analysis is enabled).",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=None,
        help="Compatibility flag; not used (landmarks are computed online when face analysis is enabled).",
    )
    parser.add_argument("--face-device", default=None, help="Device for face landmark extraction (cpu/cuda).")
    parser.add_argument("--crop-pad", type=float, default=0.08, help="Padding ratio around landmark crop.")
    parser.add_argument(
        "--no-face-analysis",
        action="store_true",
        help="Disable landmark detection + face crop pre-processing.",
    )
    return parser.parse_args()


def collect_labeled_images(dataset_root: Path) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(dataset_root)
        if not rel.parts:
            continue
        class_name = rel.parts[0].lower()
        if class_name not in CLASS_ORDER:
            continue
        label_idx = CLASS_ORDER.index(class_name)
        samples.append((path, label_idx))
    return samples


def write_confusion_csv(output_csv: Path, matrix: np.ndarray) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *EMOTION_LABELS])
        for row_idx, row in enumerate(matrix.tolist()):
            writer.writerow([EMOTION_LABELS[row_idx], *row])


def _metadata_key(path: Path, base_root: Path) -> str:
    try:
        rel = path.relative_to(base_root)
    except ValueError:
        rel = path
    return rel.as_posix()


def load_landmark_metadata(metadata_file: Path, expected_dim: int) -> dict[str, np.ndarray]:
    if expected_dim <= 0 or not metadata_file.exists():
        return {}
    table: dict[str, np.ndarray] = {}
    with metadata_file.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row.get("image_path") or "").replace("\\", "/")
            raw = row.get("landmarks") or ""
            if not key or not raw:
                continue
            try:
                coords = np.array(json.loads(raw), dtype=np.float32).reshape(-1)
            except (json.JSONDecodeError, ValueError):
                continue
            if coords.size != expected_dim:
                continue
            table[key] = coords
    return table


def save_confusion_plot(output_image: Path, matrix: np.ndarray, accuracy: float) -> None:
    output_image.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(EMOTION_LABELS)),
        yticks=np.arange(len(EMOTION_LABELS)),
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
        xlabel="Predicted",
        ylabel="True",
        title=f"Confusion Matrix (acc={accuracy*100:.2f}%)",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    max_val = max(int(matrix.max()), 1)
    threshold = max_val / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_image, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%H:%M:%S")
    device = get_device(args.device)
    LOGGER.info("Using device %s", device)

    if not args.dataset.is_dir():
        raise FileNotFoundError(f"{args.dataset} is not a directory")

    samples = collect_labeled_images(args.dataset)
    if not samples:
        raise FileNotFoundError(
            f"No labeled images found under {args.dataset}. Expected class folders: {', '.join(CLASS_ORDER)}"
        )
    LOGGER.info("Found %d labeled images in %s", len(samples), args.dataset)

    transform = build_eval_transform(args.filter)
    model, landmark_dim = build_evaluation_model(args, device)
    expects_landmarks = bool(getattr(model, "_expects_landmarks", False))
    LOGGER.info("Model expects landmarks: %s (dim=%d)", expects_landmarks, landmark_dim)
    if args.metadata_file is not None or args.metadata_root is not None:
        LOGGER.info("Ignoring --metadata-file/--metadata-root (not required by this script).")

    metadata_root = args.metadata_root or args.dataset.parent
    metadata_file = args.metadata_file or (args.dataset.parent / "metadata.csv")
    metadata_map: dict[str, np.ndarray] = {}
    missing_metadata = 0
    if expects_landmarks and metadata_file.exists():
        metadata_map = load_landmark_metadata(metadata_file, landmark_dim)
        LOGGER.info(
            "Using metadata landmarks from %s (root=%s, entries=%d).",
            metadata_file,
            metadata_root,
            len(metadata_map),
        )
    elif expects_landmarks:
        LOGGER.info("No metadata file found at %s.", metadata_file)

    aligner = None
    if expects_landmarks and not metadata_map and not args.no_face_analysis:
        import face_alignment

        face_device = args.face_device or ("cuda" if device.type == "cuda" else "cpu")
        aligner = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device=face_device,
        )
        LOGGER.info("Face analysis enabled (landmark_device=%s, crop_pad=%.3f).", face_device, args.crop_pad)
    elif args.no_face_analysis:
        LOGGER.info("Face analysis disabled via --no-face-analysis.")
    elif not expects_landmarks:
        LOGGER.info("Face analysis skipped because checkpoint does not use landmarks.")
    else:
        LOGGER.info("Face analysis skipped because landmarks are loaded from metadata.")

    confusion = np.zeros((len(EMOTION_LABELS), len(EMOTION_LABELS)), dtype=np.int64)
    skipped = 0
    for path, truth in samples:
        try:
            if expects_landmarks and metadata_map:
                from PIL import Image

                key = _metadata_key(path, metadata_root)
                landmark_vector = metadata_map.get(key)
                if landmark_vector is None:
                    missing_metadata += 1
                    landmark_vector = np.zeros((landmark_dim,), dtype=np.float32)
                pil_image = Image.open(path).convert("RGB")
            elif aligner is not None:
                pil_image, landmark_vector, *_ = prepare_face_sample(
                    path,
                    aligner,
                    crop_pad=args.crop_pad,
                    landmark_dim=landmark_dim,
                )
            else:
                from PIL import Image

                pil_image = Image.open(path).convert("RGB")
                landmark_vector = np.zeros((landmark_dim,), dtype=np.float32)

            tensor = transform(pil_image).unsqueeze(0).to(device)
            landmarks_tensor = None
            if expects_landmarks:
                landmarks_tensor = torch.from_numpy(landmark_vector).float().unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor, landmarks_tensor) if expects_landmarks else model(tensor)
                pred = int(torch.argmax(logits, dim=1).item())
            confusion[truth, pred] += 1
        except Exception as exc:  # pragma: no cover
            skipped += 1
            LOGGER.warning("Skipping %s: %s", path, exc)

    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    accuracy = (correct / total) if total > 0 else 0.0
    LOGGER.info("Evaluation complete: used=%d skipped=%d accuracy=%.2f%%", total, skipped, accuracy * 100.0)
    if expects_landmarks and metadata_map:
        LOGGER.info("Metadata landmark summary: found=%d missing=%d", total - missing_metadata, missing_metadata)

    write_confusion_csv(args.output_csv, confusion)
    save_confusion_plot(args.output_image, confusion, accuracy)
    LOGGER.info("Wrote confusion matrix CSV to %s", args.output_csv)
    LOGGER.info("Wrote confusion matrix image to %s", args.output_image)


if __name__ == "__main__":
    main()
