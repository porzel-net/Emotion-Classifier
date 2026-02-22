#!/usr/bin/env python3
"""Score a folder of images using the Neconet checkpoint and write CSV probabilities."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from helpers.neconet_helpers import (
    EMOTION_LABELS,
    apply_laplacian_to_pil,
    apply_roberts_to_pil,
    apply_sobel_to_pil,
    get_device,
    load_state_dict,
)
from train.train_emotion_classifier import Block, EmotionResNet

LOGGER = logging.getLogger(__name__)
CSV_COLUMNS = ["filepath", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]
CSV_ORDER = ["Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear"]
CLASS_ORDER = ["angry", "disgusted", "fearful", "happy", "sad", "surprised"]
DEFAULT_METADATA = Path("data/fer2013-prepared/metadata.csv")
DEFAULT_MODEL_WEIGHTS = Path("models/emotion-classifier-best.pth")

FILTER_TRANSFORMS = {
    "sobel": transforms.Lambda(apply_sobel_to_pil),
    "roberts": transforms.Lambda(apply_roberts_to_pil),
    "laplacian": transforms.Lambda(apply_laplacian_to_pil),
    "none": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score each image in a folder and emit CSV probabilities (evaluation-model checkpoints only)."
    )
    parser.add_argument("images", type=Path, help="Folder containing input images.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("predictions.csv"),
        help="CSV file to write (overwrites if it exists).",
    )
    parser.add_argument(
        "--weights",
        "--weights-dir",
        dest="weights",
        type=Path,
        default=DEFAULT_MODEL_WEIGHTS,
        help="Evaluation checkpoint path (default: models/emotion-classifier-best.pth).",
    )
    parser.add_argument("--device", default=None, help="Torch device to use (cpu/mps/cuda).")
    parser.add_argument(
        "--filter",
        choices=list(FILTER_TRANSFORMS.keys()),
        default="sobel",
        help="Preprocessing filter used during training.",
    )
    parser.add_argument("--width-multiplier", type=float, default=1.0, help="Width multiplier used during training.")
    parser.add_argument("--use-landmarks", action="store_true", help="Enable landmark input branch at inference.")
    parser.add_argument("--use-gnn", action="store_true", help="Enable GNN landmark branch at inference.")
    parser.add_argument("--gnn-hidden-dim", type=int, default=64, help="GNN hidden dim (must match training).")
    parser.add_argument("--gnn-steps", type=int, default=2, help="GNN message steps (must match training).")
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=DEFAULT_METADATA,
        help="CSV landmark metadata file used to infer/load landmark vectors.",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path("."),
        help="Root path used to resolve metadata image_path keys against input file paths.",
    )
    return parser.parse_args()


def _metadata_key(path: Path, base_root: Path) -> str:
    try:
        rel = path.relative_to(base_root)
    except ValueError:
        rel = path
    return rel.as_posix()


def load_landmark_metadata(metadata_file: Path) -> tuple[dict[str, np.ndarray], int]:
    if not metadata_file.exists():
        return {}, 0
    table: dict[str, np.ndarray] = {}
    landmark_dim = 0
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
            if coords.size == 0:
                continue
            table[key] = coords
            landmark_dim = coords.size
    return table, landmark_dim


def build_eval_transform(filter_name: str) -> transforms.Compose:
    ops: list[transforms.Transform] = [
        transforms.Resize((64, 64)),
    ]
    filter_op = FILTER_TRANSFORMS.get(filter_name)
    if filter_op is not None:
        ops.append(filter_op)
    ops.extend(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return transforms.Compose(ops)


def build_evaluation_model(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, int]:
    landmark_map: dict[str, np.ndarray] = {}
    landmark_dim = 0
    if args.use_landmarks:
        landmark_map, landmark_dim = load_landmark_metadata(args.metadata_file)
        if landmark_dim == 0:
            LOGGER.warning("Landmarks requested but no valid metadata found in %s. Using zero landmark vector.", args.metadata_file)
    model = EmotionResNet(
        Block,
        layers=(2, 2, 2, 2),
        num_classes=len(CLASS_ORDER),
        width_multiplier=args.width_multiplier,
        landmark_dim=landmark_dim if args.use_landmarks else 0,
        use_gnn=args.use_gnn,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_message_steps=args.gnn_steps,
    )
    model.to(device)
    state = load_state_dict(args.weights, device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    # Store for caller via attribute to avoid extra tuple complexity.
    model._landmark_map = landmark_map  # type: ignore[attr-defined]
    return model, landmark_dim


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    device = get_device(args.device)
    LOGGER.info("Using device %s", device)

    if not args.images.is_dir():
        raise FileNotFoundError(f"{args.images} is not a directory")
    transform = build_eval_transform(args.filter)
    model, landmark_dim = build_evaluation_model(args, device)
    landmark_map = getattr(model, "_landmark_map", {})
    LOGGER.info(
        "Loaded evaluation model (filter=%s, width=%.2f, landmarks=%s, gnn=%s, landmark_dim=%d)",
        args.filter,
        args.width_multiplier,
        args.use_landmarks,
        args.use_gnn,
        landmark_dim,
    )

    image_paths = sorted(p for p in args.images.iterdir() if p.is_file())
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.images}")

    rows = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        landmarks_tensor = None
        if landmark_dim > 0:
            key = _metadata_key(path, args.metadata_root)
            raw = landmark_map.get(key)
            if raw is None or raw.size != landmark_dim:
                landmarks_tensor = torch.zeros((1, landmark_dim), dtype=torch.float32, device=device)
            else:
                landmarks_tensor = torch.from_numpy(raw).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor, landmarks_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        rows.append((path.as_posix(), reorder_probs(probs)))

    LOGGER.info("Writing %d rows to %s", len(rows), args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for filepath, probs in rows:
            writer.writerow([filepath] + [f"{score:.2f}" for score in probs])


def reorder_probs(probs: list[float]) -> list[float]:
    order_map = {name: idx for idx, name in enumerate(EMOTION_LABELS)}
    return [probs[order_map[label]] for label in CSV_ORDER]


if __name__ == "__main__":
    main()
