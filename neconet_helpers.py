"""Shared helpers for loading the Neconet model, dataset and device selection."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from Model_architecture_Code import Block, ResNet

LOGGER = logging.getLogger(__name__)

EMOTION_LABELS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
CLASS_ORDER = ["angry", "disgusted", "fearful", "happy", "sad", "surprised"]
NECONET_PLAN = [2, 2, 2, 2]
IMAGE_SIZE = 64
NORMALIZE_MEAN = (0.5,)
NORMALIZE_STD = (0.5,)
DEFAULT_WEIGHTS = Path("Neconet_Weights3")
DEFAULT_DATASET_ROOT = Path("data/emotion-classifier-dataset")

Record = Tuple[str, int, int, float]


class EmotionFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, label, path


def filter_classes(dataset: EmotionFolderWithPaths) -> None:
    label_map = {name: idx for idx, name in enumerate(CLASS_ORDER)}
    filtered: List[tuple[str, int]] = []
    for path, original_idx in dataset.samples:
        class_name = dataset.classes[original_idx]
        if class_name in label_map:
            filtered.append((path, label_map[class_name]))
    dataset.samples = filtered
    dataset.targets = [label for _, label in filtered]
    dataset.imgs = filtered
    dataset.classes = CLASS_ORDER
    dataset.class_to_idx = label_map


def trim_dataset(dataset: EmotionFolderWithPaths, max_samples: Optional[int] = None) -> None:
    if max_samples is None or max_samples >= len(dataset.samples):
        return
    trimmed = dataset.samples[:max_samples]
    dataset.samples = trimmed
    dataset.targets = [label for _, label in trimmed]
    dataset.imgs = trimmed


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _normalize_edge(map_: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(map_, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def apply_sobel(image: np.ndarray) -> np.ndarray:
    gray = _to_grayscale(image)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return _normalize_edge(magnitude)


def apply_roberts(image: np.ndarray) -> np.ndarray:
    gray = _to_grayscale(image)
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    grad_x = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
    grad_y = cv2.filter2D(gray, cv2.CV_32F, kernel_y)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return _normalize_edge(magnitude)


def apply_laplacian(image: np.ndarray) -> np.ndarray:
    gray = _to_grayscale(image)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return _normalize_edge(np.abs(lap))


def apply_sobel_to_pil(image: Image.Image) -> Image.Image:
    array = np.array(image)
    sobel_array = apply_sobel(array)
    return Image.fromarray(sobel_array)


def apply_roberts_to_pil(image: Image.Image) -> Image.Image:
    array = np.array(image)
    roberts_array = apply_roberts(array)
    return Image.fromarray(roberts_array)


def apply_laplacian_to_pil(image: Image.Image) -> Image.Image:
    array = np.array(image)
    laplacian_array = apply_laplacian(array)
    return Image.fromarray(laplacian_array)


def build_transform(sobel: bool = False) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            *([transforms.Lambda(apply_sobel_to_pil)] if sobel else []),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


def build_loader(
    root: Path,
    split: str,
    batch_size: int = 32,
    workers: int = 2,
    max_samples: Optional[int] = None,
    sobel: bool = False,
) -> tuple[EmotionFolderWithPaths, DataLoader]:
    dataset = EmotionFolderWithPaths(root=root / split, transform=build_transform(sobel=sobel))
    filter_classes(dataset)
    trim_dataset(dataset, max_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    return dataset, loader


def _package_weights(weights_dir: Path) -> str:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    with zipfile.ZipFile(temp_file.name, "w", compression=zipfile.ZIP_STORED) as archive:
        for root, _, files in os.walk(weights_dir):
            for name in files:
                full = Path(root) / name
                rel = full.relative_to(weights_dir)
                archive.write(full, arcname=str(weights_dir.name / rel))
    return temp_file.name


def load_state_dict(weights: Path, device: torch.device):
    temp_dirs: List[Path] = []
    archive_path: Optional[str] = None
    try:
        if weights.is_file() and weights.suffix in {".pth", ".pt"}:
            return torch.load(weights, map_location=device)

        if weights.is_file():
            extracted = Path(tempfile.mkdtemp())
            temp_dirs.append(extracted)
            with zipfile.ZipFile(weights, "r") as archive:
                archive.extractall(extracted)
            root_dir = Path(tempfile.mkdtemp()) / weights.stem
            root_dir.mkdir()
            for entry in extracted.iterdir():
                shutil.move(str(entry), root_dir / entry.name)
            temp_dirs.append(root_dir)
            archive_path = _package_weights(root_dir)
        else:
            archive_path = _package_weights(weights)
        checkpoint = torch.load(archive_path, map_location=device)
    finally:
        if archive_path and os.path.exists(archive_path):
            os.remove(archive_path)
        for directory in temp_dirs:
            shutil.rmtree(directory, ignore_errors=True)
    return checkpoint


def describe_model(model: nn.Module, logger: Optional[logging.Logger] = None) -> None:
    logger = logger or LOGGER
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    modules = ", ".join(name for name, _ in model.named_children())
    logger.info(
        "Model summary: ResNet | %d total params (%d trainable) | modules=%s",
        total,
        trainable,
        modules,
    )


def build_model(
    weights: Path,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
) -> nn.Module:
    logger = logger or LOGGER
    model = ResNet(Block, NECONET_PLAN, len(EMOTION_LABELS))
    model.to(device)
    state = load_state_dict(weights, device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    describe_model(model, logger=logger)
    return model


def get_device(choice: Optional[str] = None) -> torch.device:
    if choice:
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_records(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    limit: Optional[int] = None,
) -> list[Record]:
    records: list[Record] = []
    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            conf, _ = torch.max(probs, dim=1)

            for path, truth, pred, confidence in zip(paths, labels.tolist(), preds.cpu().tolist(), conf.cpu().tolist()):
                records.append((path, truth, pred, confidence))
                if limit and len(records) >= limit:
                    return records[:limit]
    if limit:
        return records[:limit]
    return records
