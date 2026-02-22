"""Shared helpers for loading the Neconet model, dataset and device selection."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .model_architecture import Block, ResNet

LOGGER = logging.getLogger(__name__)

EMOTION_LABELS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
CLASS_ORDER = ["angry", "disgusted", "fearful", "happy", "sad", "surprised"]
NECONET_PLAN = [2, 2, 2, 2]
IMAGE_SIZE = 64
NORMALIZE_MEAN = (0.5,)
NORMALIZE_STD = (0.5,)
DEFAULT_WEIGHTS = Path("models/Neconet_Weights3.pth")
DEFAULT_DATASET_ROOT = Path("data/fer2013-prepared")

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
        "Model summary: %s | %d total params (%d trainable) | modules=%s",
        model.__class__.__name__,
        total,
        trainable,
        modules,
    )


def _normalize_state_dict(raw_state: Any) -> dict[str, torch.Tensor]:
    state = raw_state
    if isinstance(state, dict):
        if isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        elif isinstance(state.get("model_state_dict"), dict):
            state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state)!r}")
    if state and all(isinstance(key, str) and key.startswith("module.") for key in state.keys()):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def _is_emotion_resnet_state(state: dict[str, torch.Tensor]) -> bool:
    if any(key.startswith(("landmark_attention.", "gnn_branch.", "sobel_branch.")) for key in state):
        return True
    conv1 = state.get("conv1.weight")
    if conv1 is not None and conv1.ndim == 4 and conv1.shape[0] != 64:
        return True
    return "conv1.bias" not in state


def _layer4_channels(state: dict[str, torch.Tensor]) -> int:
    for key, tensor in state.items():
        if key.startswith("layer4.") and key.endswith(".conv2.weight") and tensor.ndim == 4:
            return int(tensor.shape[0])
    tensor = state.get("layer4.0.conv1.weight")
    if tensor is None or tensor.ndim != 4:
        raise KeyError("Unable to infer layer4 channel count from checkpoint.")
    return int(tensor.shape[0])


def _infer_width_multiplier(state: dict[str, torch.Tensor]) -> float:
    channel_observations: list[tuple[int, int]] = []
    key_to_base = {
        "conv1.weight": 64,
        "layer1.0.conv1.weight": 64,
        "layer2.0.conv1.weight": 128,
        "layer3.0.conv1.weight": 256,
        "layer4.0.conv1.weight": 512,
    }
    for key, base in key_to_base.items():
        tensor = state.get(key)
        if tensor is None or tensor.ndim != 4:
            continue
        channel_observations.append((int(tensor.shape[0]), base))

    if not channel_observations:
        conv1 = state.get("conv1.weight")
        if conv1 is None or conv1.ndim != 4:
            raise KeyError("Unable to infer width multiplier from checkpoint.")
        return max(0.25, float(conv1.shape[0]) / 64.0)

    low = 0.25
    high = float("inf")
    for observed, base in channel_observations:
        low = max(low, observed / float(base))
        high = min(high, (observed + 1) / float(base))

    if low < high:
        return (low + high) / 2.0

    # Fallback for inconsistent checkpoints: satisfy the strongest lower bound.
    return low + 1e-6


def _infer_emotion_resnet_kwargs(
    state: dict[str, torch.Tensor],
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    logger = logger or LOGGER
    fc_weight = state.get("fc.weight")
    if fc_weight is None:
        raise KeyError("Checkpoint is missing required key (fc.weight).")

    width_multiplier = _infer_width_multiplier(state)
    num_classes = int(fc_weight.shape[0])
    fc_input_dim = int(fc_weight.shape[1])
    backbone_dim = _layer4_channels(state)

    use_landmarks = any(key.startswith("landmark_attention.") for key in state)
    use_gnn = any(key.startswith("gnn_branch.") for key in state)
    use_sobel_branch = any(key.startswith("sobel_branch.") for key in state)

    landmark_branch_dim = 0
    branch_weight = state.get("landmark_attention.feature_branch.3.weight")
    if branch_weight is not None and branch_weight.ndim >= 1:
        landmark_branch_dim = int(branch_weight.shape[0])

    gnn_hidden_dim = 64
    gnn_weight = state.get("gnn_branch.node_proj.weight")
    if gnn_weight is not None and gnn_weight.ndim >= 1:
        gnn_hidden_dim = int(gnn_weight.shape[0])

    sobel_branch_dim = 32
    sobel_weight = state.get("sobel_branch.extractor.3.weight")
    if sobel_weight is not None and sobel_weight.ndim >= 1:
        sobel_branch_dim = int(sobel_weight.shape[0])

    landmark_dim = 0
    if use_landmarks:
        inferred = fc_input_dim - backbone_dim - landmark_branch_dim
        if use_gnn:
            inferred -= gnn_hidden_dim
        if use_sobel_branch:
            inferred -= sobel_branch_dim
        landmark_dim = max(0, int(inferred))

    logger.info(
        "Detected EmotionResNet checkpoint: width=%.2f classes=%d landmarks=%d gnn=%s gnn_hidden=%d sobel=%s sobel_dim=%d",
        width_multiplier,
        num_classes,
        landmark_dim,
        use_gnn,
        gnn_hidden_dim,
        use_sobel_branch,
        sobel_branch_dim,
    )
    return {
        "width_multiplier": width_multiplier,
        "num_classes": num_classes,
        "landmark_dim": landmark_dim,
        "use_gnn": use_gnn,
        "gnn_hidden_dim": gnn_hidden_dim,
        "use_sobel_branch": use_sobel_branch,
        "sobel_branch_dim": sobel_branch_dim,
    }


def build_model(
    weights: Path,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
) -> nn.Module:
    logger = logger or LOGGER
    state = _normalize_state_dict(load_state_dict(weights, device))

    legacy_model = ResNet(Block, NECONET_PLAN, len(EMOTION_LABELS)).to(device)
    try:
        legacy_model.load_state_dict(state, strict=True)
        legacy_model.eval()
        describe_model(legacy_model, logger=logger)
        return legacy_model
    except RuntimeError as exc:
        if not _is_emotion_resnet_state(state):
            raise RuntimeError(
                "Checkpoint is incompatible with the legacy Neconet ResNet architecture."
            ) from exc

    from train.train_emotion_classifier import Block as EvalBlock, EmotionResNet

    kwargs = _infer_emotion_resnet_kwargs(state, logger=logger)
    eval_model = EmotionResNet(
        EvalBlock,
        layers=(2, 2, 2, 2),
        num_classes=kwargs["num_classes"],
        width_multiplier=kwargs["width_multiplier"],
        landmark_dim=kwargs["landmark_dim"],
        use_gnn=kwargs["use_gnn"],
        gnn_hidden_dim=kwargs["gnn_hidden_dim"],
        gnn_message_steps=2,
        use_sobel_branch=kwargs["use_sobel_branch"],
        sobel_branch_dim=kwargs["sobel_branch_dim"],
    ).to(device)
    eval_model.load_state_dict(state, strict=True)
    eval_model.eval()
    describe_model(eval_model, logger=logger)
    return eval_model


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
