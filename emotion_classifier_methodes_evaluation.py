"""Reproduction of the lightweight Neconet ResNet with configurable preprocessing/optimization experiments."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from tqdm import tqdm

from neconet_helpers import (
    CLASS_ORDER,
    EmotionFolderWithPaths,
    apply_laplacian_to_pil,
    apply_roberts_to_pil,
    apply_sobel_to_pil,
    filter_classes,
)


# ---------- configuration constants ----------
RANDOM_SEED = 1
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 40
IMAGE_SIZE = 64
NUM_CLASSES = len(CLASS_ORDER)
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
DEFAULT_ROOT = Path("data/emotion-classifier-dataset")
DEFAULT_METADATA = DEFAULT_ROOT / "metadata.csv"
MODEL_DIR = Path("models")

FILTER_TRANSFORMS = {
    "sobel": transforms.Lambda(apply_sobel_to_pil),
    "roberts": transforms.Lambda(apply_roberts_to_pil),
    "laplacian": transforms.Lambda(apply_laplacian_to_pil),
    "none": None,
}


# ---------- model definition ----------

def conv(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = conv(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.conv2 = conv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.act(out)


class EmotionResNet(nn.Module):
    def __init__(
        self,
        block: type[Block],
        layers: Iterable[int],
        num_classes: int,
        width_multiplier: float = 1.0,
        landmark_dim: int = 0,
    ):
        super().__init__()
        self.width_multiplier = max(0.25, width_multiplier)
        base_first = 64
        scaled_first = max(1, int(base_first * self.width_multiplier))
        self.in_channels = scaled_first
        self.conv1 = nn.Conv2d(1, scaled_first, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(scaled_first)
        self.act = nn.SiLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        base_channels = [64, 128, 256, 512]
        scaled_channels = [
            max(1, int(ch * self.width_multiplier)) for ch in base_channels
        ]
        self.layer1 = self._make_layer(block, scaled_channels[0], layers[0])
        self.layer2 = self._make_layer(block, scaled_channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, scaled_channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, scaled_channels[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.35)
        self.landmark_dim = max(0, landmark_dim)
        self.fc_input_dim = scaled_channels[-1] + self.landmark_dim
        self.fc = nn.Linear(self.fc_input_dim, num_classes)

    def forward(self, x: torch.Tensor, landmarks: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        if self.landmark_dim > 0:
            if landmarks is None:
                landmarks = torch.zeros(x.size(0), self.landmark_dim, device=x.device)
            features = torch.cat([features, landmarks], dim=1)
        features = self.dropout(features)
        return self.fc(features)

    def _make_layer(self, block: type[Block], out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample: nn.Module | None = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)


# ---------- helpers & transforms ----------

def gaussian_noise(std: float) -> transforms.Lambda:
    def _add_noise(tensor: torch.Tensor) -> torch.Tensor:
        if std <= 0:
            return tensor
        noise = torch.randn_like(tensor) * std
        return torch.clamp(tensor + noise, -1.0, 1.0)

    return transforms.Lambda(_add_noise)


def build_transforms(
    filter_name: str,
    augment_rotation: bool,
    augment_scale: bool,
    augment_translation: bool,
    augment_noise: bool,
    noise_std: float,
) -> tuple[transforms.Compose, transforms.Compose]:
    if augment_scale:
        train_ops: list[transforms.Transform] = [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        ]
    else:
        train_ops = [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ]

    filter_transform = FILTER_TRANSFORMS.get(filter_name)
    if filter_transform:
        train_ops.append(filter_transform)

    if augment_rotation:
        train_ops.append(transforms.RandomRotation(15))

    if augment_translation:
        train_ops.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))

    train_ops.extend(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )

    if augment_noise:
        train_ops.append(gaussian_noise(noise_std))

    train_ops.append(transforms.Normalize((0.5,), (0.5,)))

    eval_ops: list[transforms.Transform] = []
    if augment_scale:
        eval_ops.append(transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)))
    else:
        eval_ops.append(transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)))
    if filter_transform:
        eval_ops.append(filter_transform)
    eval_ops.extend(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def _metadata_key(path: Path, base_root: Path) -> str:
    try:
        rel = path.relative_to(base_root)
    except ValueError:
        rel = path
    return rel.as_posix()


def load_landmark_metadata(metadata_file: Path, base_root: Path) -> tuple[dict[str, np.ndarray], int]:
    if not metadata_file.exists():
        logging.warning("Landmark metadata %s missing; skipping landmark features", metadata_file)
        return {}, 0

    landmarks_map: dict[str, np.ndarray] = {}
    landmark_dim = 0
    with metadata_file.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            relative_path = row.get("image_path")
            raw = row.get("landmarks")
            if not relative_path or not raw:
                continue
            try:
                coords = json.loads(raw)
            except json.JSONDecodeError:
                continue
            array = np.array(coords, dtype=np.float32).reshape(-1)
            if array.size == 0:
                continue
            landmarks_map[relative_path.replace("\\", "/")] = array
            landmark_dim = array.size
    if landmark_dim == 0:
        logging.warning("No landmarks parsed from %s", metadata_file)
    return landmarks_map, landmark_dim


class LandmarkAwareEmotionFolder(EmotionFolderWithPaths):
    def __init__(
        self,
        root: str | Path,
        transform: Optional[transforms.Compose],
        metadata_map: dict[str, np.ndarray],
        metadata_root: Path,
        landmark_dim: int,
    ):
        super().__init__(root=root, transform=transform)
        self.metadata_map = metadata_map
        self.metadata_root = metadata_root
        self.landmark_dim = max(0, landmark_dim)

    def _lookup_landmarks(self, path: str) -> torch.Tensor:
        key = _metadata_key(Path(path), self.metadata_root)
        vector = self.metadata_map.get(key)
        if vector is None or vector.size == 0:
            return torch.zeros(self.landmark_dim, dtype=torch.float32)
        return torch.from_numpy(vector)

    def __getitem__(self, index: int):
        image, label, path = super().__getitem__(index)
        landmarks = (
            self._lookup_landmarks(path)
            if self.landmark_dim > 0
            else torch.zeros(0, dtype=torch.float32)
        )
        return image, label, landmarks


def _count_targets(dataset: EmotionFolderWithPaths | Subset) -> Counter[int]:
    if isinstance(dataset, Subset):
        return Counter(dataset.dataset.targets[idx] for idx in dataset.indices)
    return Counter(dataset.targets)


def _balanced_indices(dataset: EmotionFolderWithPaths) -> list[int]:
    buckets: dict[int, list[int]] = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        buckets[label].append(idx)
    min_count = min(len(bucket) for bucket in buckets.values())
    rng = random.Random(RANDOM_SEED)
    selected: list[int] = []
    for label in range(NUM_CLASSES):
        choices = buckets[label]
        rng.shuffle(choices)
        selected.extend(choices[:min_count])
    rng.shuffle(selected)
    return selected


def _split_dataset(dataset: EmotionFolderWithPaths | Subset, split: float) -> tuple[EmotionFolderWithPaths | Subset, Optional[Subset]]:
    if split <= 0.0:
        return dataset, None
    split = min(max(split, 0.01), 0.5)
    val_size = int(len(dataset) * split)
    if val_size == 0 or len(dataset) - val_size == 0:
        return dataset, None
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset


def build_dataloaders(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, Optional[DataLoader], DataLoader, Counter[int], int]:
    train_transform, eval_transform = build_transforms(
        args.filter,
        args.augment_rotation,
        args.augment_scale,
        args.augment_translation,
        args.augment_noise,
        args.noise_std,
    )

    metadata_map: dict[str, np.ndarray] = {}
    landmark_dim = 0
    if args.use_landmarks:
        metadata_map, landmark_dim = load_landmark_metadata(args.metadata_file, args.data_root)
    use_landmarks = args.use_landmarks and landmark_dim > 0

    if use_landmarks:
        train_dataset = LandmarkAwareEmotionFolder(
            root=args.data_root / "train",
            transform=train_transform,
            metadata_map=metadata_map,
            metadata_root=args.data_root,
            landmark_dim=landmark_dim,
        )
    else:
        train_dataset = EmotionFolderWithPaths(root=args.data_root / "train", transform=train_transform)
    filter_classes(train_dataset)

    if args.sampling == "undersample":
        sampled = Subset(train_dataset, _balanced_indices(train_dataset))
    else:
        sampled = train_dataset

    sampled_counts = _count_targets(sampled)

    train_subset, val_subset = _split_dataset(sampled, args.val_split)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        dataset=train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = (
        DataLoader(
            dataset=val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )
        if val_subset is not None
        else None
    )

    if use_landmarks:
        test_dataset = LandmarkAwareEmotionFolder(
            root=args.data_root / "test",
            transform=eval_transform,
            metadata_map=metadata_map,
            metadata_root=args.data_root,
            landmark_dim=landmark_dim,
        )
    else:
        test_dataset = EmotionFolderWithPaths(root=args.data_root / "test", transform=eval_transform)
    filter_classes(test_dataset)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, sampled_counts, landmark_dim


def compute_class_weights(counts: Counter[int]) -> Optional[torch.Tensor]:
    if not counts:
        return None
    total = sum(counts.values())
    weights = []
    for cls in range(NUM_CLASSES):
        cls_count = counts.get(cls, 1)
        weights.append(total / (NUM_CLASSES * cls_count))
    return torch.tensor(weights, dtype=torch.float32)


def build_loss_function(args: argparse.Namespace, counts: Counter[int]) -> nn.Module:
    if args.loss == "mse":
        return nn.MSELoss()
    weight = None
    if args.sampling == "loss-weight":
        weight = compute_class_weights(counts)
    return nn.CrossEntropyLoss(weight=weight)


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    if isinstance(criterion, nn.MSELoss):
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, NUM_CLASSES).float()
        return criterion(probs, one_hot)
    return criterion(logits, targets)


def _prepare_batch(batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]):
    inputs, targets = batch[:2]
    landmarks = batch[2] if len(batch) > 2 else None
    if landmarks is not None and not isinstance(landmarks, torch.Tensor):
        landmarks = None
    return inputs, targets, landmarks


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    num_epochs: int,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    progress = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for batch in progress:
        inputs, targets, landmarks = _prepare_batch(batch)
        inputs = inputs.to(device)
        targets = targets.to(device)
        if landmarks is not None:
            landmarks = landmarks.to(device)

        logits = model(inputs, landmarks)
        loss = compute_loss(logits, targets, criterion)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.item() * inputs.size(0)
        total_loss += batch_loss
        total_samples += inputs.size(0)
        progress.set_postfix(loss=loss.item())

    return total_loss / total_samples if total_samples else 0.0


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_targets: list[int] = []
    all_preds: list[int] = []

    with torch.no_grad():
        for batch in loader:
            inputs, targets, landmarks = _prepare_batch(batch)
            inputs = inputs.to(device)
            targets = targets.to(device)
            if landmarks is not None:
                landmarks = landmarks.to(device)

            logits = model(inputs, landmarks)
            loss = compute_loss(logits, targets, criterion)

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            preds = torch.argmax(logits, dim=1)
            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    loss_value = running_loss / total_samples if total_samples else 0.0
    accuracy = accuracy_score(all_targets, all_preds)
    precision = macro_precision(all_targets, all_preds, NUM_CLASSES)
    recall = macro_recall(all_targets, all_preds, NUM_CLASSES)
    f1 = macro_f1(all_targets, all_preds, NUM_CLASSES)

    return {
        "loss": loss_value,
        "accuracy": accuracy * 100,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def accuracy_score(true_labels: list[int], pred_labels: list[int]) -> float:
    if not true_labels:
        return 0.0
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    return correct / len(true_labels)


def macro_precision(true_labels: list[int], pred_labels: list[int], num_classes: int) -> float:
    total_precision = 0.0
    for cls in range(num_classes):
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if p == cls and t == cls)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if p == cls and t != cls)
        denom = tp + fp
        total_precision += tp / denom if denom else 0.0
    return total_precision / num_classes


def macro_recall(true_labels: list[int], pred_labels: list[int], num_classes: int) -> float:
    total_recall = 0.0
    for cls in range(num_classes):
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)
        denom = tp + fn
        total_recall += tp / denom if denom else 0.0
    return total_recall / num_classes


def macro_f1(true_labels: list[int], pred_labels: list[int], num_classes: int) -> float:
    total_f1 = 0.0
    for cls in range(num_classes):
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if p == cls and t != cls)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        total_f1 += 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return total_f1 / num_classes


def log_model_summary(model: nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Model size: %d total params (%d trainable).", total, trainable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the lightweight Neconet ResNet variants.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT, help="Path to emotion-classifier-dataset")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument(
        "--scheduler",
        choices=["cosine", "plateau"],
        default="cosine",
        help="What learning-rate scheduler to use.",
    )
    parser.add_argument("--plateau-factor", type=float, default=0.5, help="ReduceLROnPlateau factor.")
    parser.add_argument("--plateau-patience", type=int, default=2, help="ReduceLROnPlateau patience.")
    parser.add_argument("--plateau-min-lr", type=float, default=1e-6, help="ReduceLROnPlateau minimum LR.")
    parser.add_argument("--sampling", choices=["original", "undersample", "loss-weight"], default="original")
    parser.add_argument(
        "--filter",
        choices=list(FILTER_TRANSFORMS.keys()),
        default="sobel",
        help="Edge filter that is applied before grayscale conversion.",
    )
    parser.add_argument("--augment-rotation", action="store_true", help="Add rotation augmentation.")
    parser.add_argument("--augment-scale", action="store_true", help="Add random resized crop (scale) augmentation.")
    parser.add_argument("--augment-translation", action="store_true", help="Add translation augmentation via RandomAffine.")
    parser.add_argument("--augment-noise", action="store_true", help="Add Gaussian noise to tensors.")
    parser.add_argument("--noise-std", type=float, default=0.02, help="Standard deviation for Gaussian noise augmentation.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of train set held out for validation.")
    parser.add_argument("--loss", choices=["ce", "mse"], default="ce", help="Loss function used during training.")
    parser.add_argument("--use-landmarks", action="store_true", help="Augment batches with metadata landmarks.")
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=DEFAULT_METADATA,
        help="CSV containing landmark annotations.",
    )
    parser.add_argument("--device", type=str, default=None, help="Override training device (cuda/cpu/mps).")
    parser.add_argument(
        "--width-multiplier",
        type=float,
        default=1.0,
        help="Scale factor (<1 for smaller models, >1 for wider).",
    )
    return parser.parse_args()


def create_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau:
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.plateau_min_lr,
        )
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, args.epochs), eta_min=1e-6
    )


def _describe_counts(counts: Counter[int]) -> str:
    return ", ".join(f"{CLASS_ORDER[cls]}={counts.get(cls, 0)}" for cls in range(NUM_CLASSES))


def _loader_summary(name: str, loader: DataLoader) -> None:
    try:
        total = len(loader.dataset)
    except AttributeError:
        total = sum(batch[0].size(0) for batch in loader)
    logging.info("%s loader: %d batches, %d samples", name, len(loader), total)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = torch.device(args.device) if args.device else DEFAULT_DEVICE
    torch.manual_seed(RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(RANDOM_SEED)

    logging.info(
        "Loading data from %s (filter=%s, sampling=%s, width=%.2f, landmarks=%s)",
        args.data_root,
        args.filter,
        args.sampling,
        args.width_multiplier,
        args.use_landmarks,
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader, class_counts, landmark_dim = build_dataloaders(args, device)
    logging.info("Class counts after sampling: %s", _describe_counts(class_counts))
    _loader_summary("Train", train_loader)
    if val_loader:
        _loader_summary("Validation", val_loader)
    _loader_summary("Test", test_loader)

    model = EmotionResNet(
        Block,
        [2, 2, 2, 2],
        NUM_CLASSES,
        width_multiplier=args.width_multiplier,
        landmark_dim=landmark_dim if args.use_landmarks else 0,
    ).to(device)
    log_model_summary(model)

    criterion = build_loss_function(args, class_counts)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, args)

    last_val_metrics: dict[str, float] = {}
    test_metrics: dict[str, float] = {}
    logging.info(
        "Training for %d epochs with scheduler=%s, loss=%s",
        args.epochs,
        args.scheduler,
        args.loss,
    )
    best_val_accuracy = -1.0
    for epoch in range(args.epochs):
        logging.info("Epoch %d/%d — training", epoch + 1, args.epochs)
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch, args.epochs, device)
        if args.scheduler == "cosine":
            scheduler.step()

        train_metrics = evaluate(model, train_loader, criterion, device)
        logging.info("Evaluating training split after epoch %d", epoch + 1)
        val_metrics = evaluate(model, val_loader, criterion, device) if val_loader else {}
        logging.info("Evaluating test split after epoch %d", epoch + 1)
        test_metrics = evaluate(model, test_loader, criterion, device)

        if args.scheduler == "plateau":
            monitor = val_metrics or test_metrics
            scheduler.step(monitor.get("loss", 0.0))
            logging.info(
                "ReduceLROnPlateau triggered with loss=%.4f -> lr=%.6f",
                monitor.get("loss", 0.0),
                optimizer.param_groups[0]["lr"],
            )

        last_val_metrics = val_metrics
        logging.info(
            "Epoch %d/%d | train loss=%.4f acc=%.2f%% f1=%.4f | test loss=%.4f acc=%.2f%% f1=%.4f",
            epoch + 1,
            args.epochs,
            train_loss,
            train_metrics["accuracy"],
            train_metrics["f1"],
            test_metrics["loss"],
            test_metrics["accuracy"],
            test_metrics["f1"],
        )
        if val_metrics:
            logging.info(
                "Validation | loss=%.4f acc=%.2f%% f1=%.4f",
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["f1"],
            )
            val_acc = val_metrics["accuracy"]
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                checkpoint = MODEL_DIR / f"emotion-classifier-step{epoch+1}.pth"
                torch.save(model.state_dict(), checkpoint)
                logging.info("Saved improved model to %s (val acc %.2f%%)", checkpoint, val_acc)

    logging.info(
        "Final evaluation | Precision=%.4f Recall=%.4f Macro-F1=%.4f",
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["f1"],
    )
    if last_val_metrics:
        logging.info(
            "Final validation | Precision=%.4f Recall=%.4f Macro-F1=%.4f",
            last_val_metrics["precision"],
            last_val_metrics["recall"],
            last_val_metrics["f1"],
        )


if __name__ == "__main__":
    main()
