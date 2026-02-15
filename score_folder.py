#!/usr/bin/env python3
"""Score a folder of images using the Neconet checkpoint and write CSV probabilities."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import torch
from PIL import Image

from neconet_helpers import (
    DEFAULT_WEIGHTS,
    EMOTION_LABELS,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    build_model,
    build_transform,
    get_device,
)

LOGGER = logging.getLogger(__name__)
CSV_COLUMNS = ["filepath", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]
CSV_ORDER = ["Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score each image in a folder and emit CSV with Neconet probabilities.")
    parser.add_argument("images", type=Path, help="Folder containing input images.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("predictions.csv"),
        help="CSV file to write (overwrites if it exists).",
    )
    parser.add_argument("--weights", "--weights-dir", dest="weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default=None, help="Torch device to use (cpu/mps/cuda).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    device = get_device(args.device)
    LOGGER.info("Using device %s", device)

    if not args.images.is_dir():
        raise FileNotFoundError(f"{args.images} is not a directory")
    transform = build_transform()

    model = build_model(args.weights, device, logger=LOGGER)

    image_paths = sorted(p for p in args.images.iterdir() if p.is_file())
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.images}")

    rows = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
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
