"""Shared helpers for loading the cropped face landmark dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
from PIL import Image

DEFAULT_SIZE: Tuple[int, int] = (64, 64)


def _parse_csv(csv_path: Path) -> Iterable[Tuple[str, Sequence[float]]]:
    with csv_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header
        for row in reader:
            if not row or not row[0]:
                continue
            coords = [float(value) for value in row[1:] if value]
            yield row[0], coords


def load_landmarks(
    csv_path: Path,
    image_root: Path,
    target_size: Tuple[int, int] = DEFAULT_SIZE,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load face images and scale their landmarks to the target resolution."""

    images: list[np.ndarray] = []
    landmarks: list[np.ndarray] = []
    for idx, (file_name, coords) in enumerate(_parse_csv(csv_path)):
        if max_samples is not None and idx >= max_samples:
            break
        image_path = image_root / file_name
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            orig_width, orig_height = img.size
            if orig_width == 0 or orig_height == 0:
                raise ValueError(f"Unexpected size for {image_path}: {img.size}")
            img = img.resize(target_size, Image.BILINEAR)
            image_arr = np.asarray(img, dtype=np.float32) / 255.0
        scale_x = target_size[0] / orig_width
        scale_y = target_size[1] / orig_height
        coord_array = np.array(coords, dtype=np.float32)
        coord_array[0::2] *= scale_x
        coord_array[1::2] *= scale_y
        images.append(image_arr)
        landmarks.append(coord_array)

    if not images:
        raise ValueError("No samples were loaded from the annotation CSV.")

    return np.stack(images), np.stack(landmarks)
