"""Reproduction of the lightweight Neconet ResNet with configurable preprocessing/optimization experiments."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from helpers.neconet_helpers import (
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
DEFAULT_ROOT = Path("data/fer2013-prepared")
MODEL_DIR = Path("models")
BEST_CHECKPOINT = MODEL_DIR / "emotion-classifier-best.pth"
DATASET_PRESETS = {
    "fer2013": Path("data/fer2013-prepared"),
    "affectnet": Path("data/affectnet-yolo-format-prepared"),
}

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


class LandmarkHeatmapGenerator(nn.Module):
    def __init__(
        self,
        num_landmarks: int,
        image_size: int,
        sigmas: Iterable[float] = (1.0, 2.0, 4.0),
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.num_landmarks = max(1, num_landmarks)
        self.sigmas = tuple(sigmas)

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        if landmarks.numel() == 0:
            return torch.zeros(
                landmarks.size(0),
                len(self.sigmas),
                self.image_size,
                self.image_size,
                device=landmarks.device,
                dtype=landmarks.dtype,
            )

        coords = landmarks.view(landmarks.size(0), -1, 2)
        coords = coords.clamp(min=0.0, max=float(self.image_size - 1))
        xs = torch.arange(self.image_size, device=coords.device, dtype=coords.dtype).view(1, 1, 1, self.image_size)
        ys = torch.arange(self.image_size, device=coords.device, dtype=coords.dtype).view(1, 1, self.image_size, 1)
        x_coords = coords[..., 0].unsqueeze(-1).unsqueeze(-1)
        y_coords = coords[..., 1].unsqueeze(-1).unsqueeze(-1)

        heatmaps: list[torch.Tensor] = []
        for sigma in self.sigmas:
            dist_sq = (xs - x_coords) ** 2 + (ys - y_coords) ** 2
            gauss = torch.exp(-dist_sq / (2 * sigma * sigma + 1e-6))
            heatmaps.append(torch.sum(gauss, dim=1, keepdim=True))

        stacked = torch.cat(heatmaps, dim=1)
        return stacked.clamp_max(1.0)


class LandmarkAttentionFusion(nn.Module):
    def __init__(self, num_heatmaps: int, image_channels: int = 1, base_channels: int = 16):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(image_channels + num_heatmaps, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor, heatmaps: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([image, heatmaps], dim=1)
        attention = self.fusion(fused)
        return image * (1.0 + attention)


class LandmarkGuidedAttention(nn.Module):
    def __init__(
        self,
        landmark_dim: int,
        image_size: int,
        image_channels: int = 1,
        base_channels: int = 16,
        feature_dim: int = 32,
    ):
        super().__init__()
        self.num_landmarks = max(1, landmark_dim // 2)
        self.generator = LandmarkHeatmapGenerator(self.num_landmarks, image_size)
        self.heatmap_channels = len(self.generator.sigmas)
        self.attention = LandmarkAttentionFusion(
            num_heatmaps=self.heatmap_channels,
            image_channels=image_channels,
            base_channels=base_channels,
        )
        self.feature_dim = max(0, feature_dim)
        if self.feature_dim > 0:
            fusion_channels = image_channels + self.heatmap_channels
            self.feature_branch = nn.Sequential(
                nn.Conv2d(fusion_channels, base_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.SiLU(),
                nn.Conv2d(base_channels, self.feature_dim, kernel_size=1),
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.feature_branch = None
            self.pool = None

    def forward(self, image: torch.Tensor, landmarks: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if landmarks is None or landmarks.numel() == 0:
            branch_features = (
                torch.zeros(
                    image.size(0),
                    self.feature_dim,
                    device=image.device,
                    dtype=image.dtype,
                )
                if self.feature_branch is not None
                else None
            )
            return image, branch_features

        heatmaps = self.generator(landmarks)
        attended = self.attention(image, heatmaps)
        branch_features = None
        if self.feature_branch is not None and self.pool is not None:
            fused = torch.cat([image, heatmaps], dim=1)
            branch_features = self.pool(self.feature_branch(fused)).view(image.size(0), -1)
        return attended, branch_features


class LandmarkGNNBranch(nn.Module):
    def __init__(self, landmark_dim: int, hidden_dim: int = 64, message_steps: int = 2) -> None:
        super().__init__()
        self.landmark_dim = max(0, landmark_dim)
        self.hidden_dim = max(1, hidden_dim)
        self.message_steps = max(1, message_steps)

        self.node_proj = nn.Linear(2, self.hidden_dim)
        self.message_lin = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.update_lin = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, landmarks: torch.Tensor | None, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        if landmarks is None or landmarks.numel() == 0:
            target_device = device or (landmarks.device if landmarks is not None else torch.device("cpu"))
            return torch.zeros(batch_size, self.hidden_dim, device=target_device)

        nodes = landmarks.view(batch_size, -1, 2)
        h = self.node_proj(nodes)
        for _ in range(self.message_steps):
            messages = F.silu(self.message_lin(h))
            aggregated = messages.mean(dim=1, keepdim=True)
            expanded = aggregated.expand(-1, nodes.size(1), -1)
            h = F.silu(self.update_lin(torch.cat([h, expanded], dim=-1)))
        pooled = self.pool(h.transpose(1, 2)).view(batch_size, self.hidden_dim)
        return pooled




class SobelFeatureBranch(nn.Module):
    def __init__(self, feature_dim: int = 32, base_channels: int = 16) -> None:
        super().__init__()
        self.feature_dim = max(1, feature_dim)
        self.extractor = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, self.feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)
        self.register_buffer("kernel_x", sobel_x)
        self.register_buffer("kernel_y", sobel_y)

    def _sobel_map(self, image: torch.Tensor) -> torch.Tensor:
        image01 = (image + 1.0) * 0.5
        grad_x = F.conv2d(image01, self.kernel_x, padding=1)
        grad_y = F.conv2d(image01, self.kernel_y, padding=1)
        magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        max_per_sample = magnitude.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return magnitude / max_per_sample

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        sobel_map = self._sobel_map(image)
        features = self.extractor(sobel_map)
        return torch.flatten(features, 1)


class EmotionResNet(nn.Module):
    def __init__(
        self,
        block: type[Block],
        layers: Iterable[int],
        num_classes: int,
        width_multiplier: float = 1.0,
        dropout: float = 0.35,
        landmark_dim: int = 0,
        use_gnn: bool = False,
        gnn_hidden_dim: int = 64,
        gnn_message_steps: int = 2,
        use_sobel_branch: bool = False,
        sobel_branch_dim: int = 32,
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
        dropout = float(dropout)
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        self.dropout = nn.Dropout(p=dropout)
        self.landmark_dim = max(0, landmark_dim)
        self.landmark_attention: LandmarkGuidedAttention | None = None
        self.landmark_branch_dim = 0
        if self.landmark_dim > 0:
            self.landmark_attention = LandmarkGuidedAttention(self.landmark_dim, IMAGE_SIZE)
            self.landmark_branch_dim = self.landmark_attention.feature_dim
        self.use_gnn = use_gnn and self.landmark_dim > 0
        self.gnn_branch_dim = 0
        self.gnn_branch: LandmarkGNNBranch | None = None
        if self.use_gnn:
            self.gnn_branch = LandmarkGNNBranch(
                self.landmark_dim,
                hidden_dim=gnn_hidden_dim,
                message_steps=gnn_message_steps,
            )
            self.gnn_branch_dim = self.gnn_branch.hidden_dim
        self.fc_input_dim = scaled_channels[-1]
        if self.landmark_dim > 0:
            self.fc_input_dim += self.landmark_dim
        if self.landmark_branch_dim:
            self.fc_input_dim += self.landmark_branch_dim
        if self.gnn_branch_dim:
            self.fc_input_dim += self.gnn_branch_dim
        self.sobel_branch: SobelFeatureBranch | None = None
        self.sobel_branch_dim = 0
        if use_sobel_branch:
            self.sobel_branch = SobelFeatureBranch(feature_dim=sobel_branch_dim)
            self.sobel_branch_dim = self.sobel_branch.feature_dim
            self.fc_input_dim += self.sobel_branch_dim
        self.fc = nn.Linear(self.fc_input_dim, num_classes)

    def forward(self, x: torch.Tensor, landmarks: Optional[torch.Tensor] = None) -> torch.Tensor:
        sobel_features: torch.Tensor | None = None
        if self.sobel_branch is not None:
            sobel_features = self.sobel_branch(x)
        branch_features: torch.Tensor | None = None
        if self.landmark_attention is not None:
            x, branch_features = self.landmark_attention(x, landmarks)
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
        if branch_features is not None:
            features = torch.cat([features, branch_features], dim=1)
        if self.gnn_branch is not None:
            gnn_features = self.gnn_branch(landmarks, x.size(0), device=x.device)
            features = torch.cat([features, gnn_features], dim=1)
        if sobel_features is not None:
            features = torch.cat([features, sobel_features], dim=1)
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
    augment_perspective: bool,
    augment_erasing: bool,
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

    if augment_perspective:
        train_ops.append(transforms.RandomPerspective(distortion_scale=0.18, p=0.4))
        train_ops.append(transforms.RandomAffine(degrees=0, shear=(-8, 8, -4, 4)))

    train_ops.extend(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )

    if augment_erasing:
        train_ops.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0.0))

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
        *,
        filter_name: str,
        augment_rotation: bool,
        augment_scale: bool,
        augment_translation: bool,
        augment_perspective: bool,
        augment_erasing: bool,
        augment_noise: bool,
        noise_std: float,
        train_mode: bool,
    ):
        super().__init__(root=root, transform=transform)
        self.metadata_map = metadata_map
        self.metadata_root = metadata_root
        self.landmark_dim = max(0, landmark_dim)
        self.filter_transform = FILTER_TRANSFORMS.get(filter_name)
        self.augment_rotation = augment_rotation and train_mode
        self.augment_scale = augment_scale and train_mode
        self.augment_translation = augment_translation and train_mode
        self.augment_perspective = augment_perspective and train_mode
        self.augment_erasing = augment_erasing and train_mode
        self.augment_noise = augment_noise and train_mode
        self.noise_std = noise_std
        self.train_mode = train_mode

    def _lookup_landmarks(self, path: str) -> torch.Tensor:
        key = _metadata_key(Path(path), self.metadata_root)
        vector = self.metadata_map.get(key)
        if vector is None or vector.size == 0 or vector.size != self.landmark_dim:
            return torch.zeros(self.landmark_dim, dtype=torch.float32)
        return torch.from_numpy(vector)

    @staticmethod
    def _to_pixel_landmarks(landmarks: torch.Tensor, width: int, height: int) -> tuple[np.ndarray, bool]:
        if landmarks.numel() == 0:
            return np.zeros((0, 2), dtype=np.float32), True
        coords = landmarks.view(-1, 2).cpu().numpy().astype(np.float32)
        is_normalized = bool(np.max(np.abs(coords)) <= 1.5)
        if is_normalized:
            coords[:, 0] *= float(width)
            coords[:, 1] *= float(height)
        return coords, is_normalized

    @staticmethod
    def _from_pixel_landmarks(coords: np.ndarray, normalized: bool, width: int, height: int) -> torch.Tensor:
        if coords.size == 0:
            return torch.zeros(0, dtype=torch.float32)
        out = coords.copy()
        if normalized:
            out[:, 0] /= max(float(width), 1.0)
            out[:, 1] /= max(float(height), 1.0)
        out[:, 0] = np.clip(out[:, 0], 0.0, 1.0 if normalized else float(width - 1))
        out[:, 1] = np.clip(out[:, 1], 0.0, 1.0 if normalized else float(height - 1))
        return torch.from_numpy(out.reshape(-1).astype(np.float32))

    @staticmethod
    def _apply_homography(coords: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        if coords.size == 0:
            return coords
        ones = np.ones((coords.shape[0], 1), dtype=np.float32)
        homo = np.concatenate([coords, ones], axis=1)
        transformed = homo @ matrix.T
        denom = np.clip(transformed[:, 2:3], 1e-6, None)
        return transformed[:, :2] / denom

    @staticmethod
    def _build_perspective_matrix(src: list[list[float]], dst: list[list[float]]) -> np.ndarray:
        src_pts = np.array(src, dtype=np.float32)
        dst_pts = np.array(dst, dtype=np.float32)
        rows = []
        values = []
        for (x, y), (u, v) in zip(src_pts, dst_pts):
            rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
            rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
            values.extend([u, v])
        a = np.asarray(rows, dtype=np.float32)
        b = np.asarray(values, dtype=np.float32)
        params, *_ = np.linalg.lstsq(a, b, rcond=None)
        h = np.array(
            [
                [params[0], params[1], params[2]],
                [params[3], params[4], params[5]],
                [params[6], params[7], 1.0],
            ],
            dtype=np.float32,
        )
        return h

    @staticmethod
    def _rotate(coords: np.ndarray, angle_deg: float, width: int, height: int) -> np.ndarray:
        if coords.size == 0:
            return coords
        theta = math.radians(angle_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        shifted = coords - np.array([cx, cy], dtype=np.float32)
        rot = np.empty_like(shifted)
        rot[:, 0] = shifted[:, 0] * cos_t - shifted[:, 1] * sin_t
        rot[:, 1] = shifted[:, 0] * sin_t + shifted[:, 1] * cos_t
        return rot + np.array([cx, cy], dtype=np.float32)

    @staticmethod
    def _shear(coords: np.ndarray, shear_x_deg: float, shear_y_deg: float) -> np.ndarray:
        if coords.size == 0:
            return coords
        shx = math.tan(math.radians(shear_x_deg))
        shy = math.tan(math.radians(shear_y_deg))
        out = np.empty_like(coords)
        out[:, 0] = coords[:, 0] + shx * coords[:, 1]
        out[:, 1] = coords[:, 1] + shy * coords[:, 0]
        return out

    def _apply_shared_transforms(self, image: Image.Image, landmarks: torch.Tensor) -> tuple[Image.Image, torch.Tensor]:
        width, height = image.size
        coords, was_normalized = self._to_pixel_landmarks(landmarks, width, height)

        if self.augment_scale:
            top, left, crop_h, crop_w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
            )
            image = TF.resized_crop(
                image,
                top=top,
                left=left,
                height=crop_h,
                width=crop_w,
                size=[IMAGE_SIZE, IMAGE_SIZE],
            )
            if coords.size:
                coords[:, 0] = (coords[:, 0] - float(left)) * (IMAGE_SIZE / float(crop_w))
                coords[:, 1] = (coords[:, 1] - float(top)) * (IMAGE_SIZE / float(crop_h))
        else:
            if (width, height) != (IMAGE_SIZE, IMAGE_SIZE):
                image = TF.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
                if coords.size:
                    coords[:, 0] *= IMAGE_SIZE / float(width)
                    coords[:, 1] *= IMAGE_SIZE / float(height)

        width, height = image.size

        if self.augment_rotation:
            angle = float(torch.empty(1).uniform_(-15.0, 15.0).item())
            image = TF.rotate(image, angle=angle)
            coords = self._rotate(coords, angle_deg=angle, width=width, height=height)

        if self.augment_translation:
            dx = int(round(float(torch.empty(1).uniform_(-0.1, 0.1).item() * width)))
            dy = int(round(float(torch.empty(1).uniform_(-0.1, 0.1).item() * height)))
            image = TF.affine(image, angle=0.0, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0])
            if coords.size:
                coords[:, 0] += float(dx)
                coords[:, 1] += float(dy)

        if self.augment_perspective:
            if random.random() < 0.4:
                startpoints, endpoints = transforms.RandomPerspective.get_params(width, height, 0.18)
                image = TF.perspective(image, startpoints=startpoints, endpoints=endpoints)
                matrix = self._build_perspective_matrix(startpoints, endpoints)
                coords = self._apply_homography(coords, matrix)

            shear_x = float(torch.empty(1).uniform_(-8.0, 8.0).item())
            shear_y = float(torch.empty(1).uniform_(-4.0, 4.0).item())
            image = TF.affine(
                image,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[shear_x, shear_y],
                center=[0.0, 0.0],
            )
            coords = self._shear(coords, shear_x_deg=shear_x, shear_y_deg=shear_y)

        if self.train_mode and random.random() < 0.5:
            image = TF.hflip(image)
            if coords.size:
                coords[:, 0] = (width - 1) - coords[:, 0]

        landmarks_out = self._from_pixel_landmarks(coords, normalized=was_normalized, width=width, height=height)
        return image, landmarks_out

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = self.loader(path)
        landmarks = (
            self._lookup_landmarks(path)
            if self.landmark_dim > 0
            else torch.zeros(0, dtype=torch.float32)
        )
        image, landmarks = self._apply_shared_transforms(image, landmarks)
        if self.filter_transform:
            image = self.filter_transform(image)
        image = TF.to_grayscale(image, num_output_channels=1)
        image = TF.to_tensor(image)
        if self.augment_erasing:
            image = transforms.RandomErasing(
                p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0.0
            )(image)
        if self.augment_noise:
            image = gaussian_noise(self.noise_std)(image)
        image = TF.normalize(image, mean=(0.5,), std=(0.5,))
        landmarks = (
            landmarks
            if self.landmark_dim > 0
            else torch.zeros(0, dtype=torch.float32)
        )
        return image, label, landmarks


def resolve_training_roots(args: argparse.Namespace) -> list[Path]:
    if args.datasets:
        roots = [DATASET_PRESETS[name] for name in args.datasets]
    else:
        roots = [args.data_root, *args.extra_data_roots]

    unique_roots: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        unique_roots.append(root)
    return unique_roots


def _dataset_targets(dataset: Dataset | Subset) -> list[int]:
    if isinstance(dataset, Subset):
        base_targets = _dataset_targets(dataset.dataset)
        return [base_targets[idx] for idx in dataset.indices]
    if isinstance(dataset, ConcatDataset):
        all_targets: list[int] = []
        for child in dataset.datasets:
            all_targets.extend(_dataset_targets(child))
        return all_targets

    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise AttributeError("Dataset does not expose targets for class balancing/counting.")
    return [int(label) for label in targets]


def _count_targets(dataset: Dataset | Subset) -> Counter[int]:
    return Counter(_dataset_targets(dataset))


def _balanced_indices(dataset: Dataset) -> list[int]:
    targets = _dataset_targets(dataset)
    buckets: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(targets):
        if 0 <= label < NUM_CLASSES:
            buckets[label].append(idx)
    if not buckets:
        return []

    min_count = min(len(bucket) for bucket in buckets.values())
    rng = random.Random(RANDOM_SEED)
    selected: list[int] = []
    for label in range(NUM_CLASSES):
        choices = buckets.get(label, [])
        if not choices:
            continue
        rng.shuffle(choices)
        selected.extend(choices[:min_count])
    rng.shuffle(selected)
    return selected


def _split_dataset(dataset: Dataset, split: float) -> tuple[Dataset, Optional[Subset]]:
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


def _build_split_dataset(
    root: Path,
    split: str,
    transform: Optional[transforms.Compose],
    use_landmarks: bool,
    metadata_map: dict[str, np.ndarray],
    landmark_dim: int,
    args: argparse.Namespace,
    *,
    train_mode: bool,
) -> Dataset:
    split_root = root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Dataset split folder not found: {split_root}")

    if use_landmarks:
        dataset: Dataset = LandmarkAwareEmotionFolder(
            root=split_root,
            transform=None,
            metadata_map=metadata_map,
            metadata_root=root,
            landmark_dim=landmark_dim,
            filter_name=args.filter,
            augment_rotation=args.augment_rotation if train_mode else False,
            augment_scale=args.augment_scale if train_mode else False,
            augment_translation=args.augment_translation if train_mode else False,
            augment_perspective=args.augment_perspective if train_mode else False,
            augment_erasing=args.augment_erasing if train_mode else False,
            augment_noise=args.augment_noise if train_mode else False,
            noise_std=args.noise_std,
            train_mode=train_mode,
        )
    else:
        if transform is None:
            raise ValueError("transform must not be None when landmarks are disabled.")
        dataset = EmotionFolderWithPaths(root=split_root, transform=transform)

    filter_classes(dataset)
    return dataset


def build_dataloaders(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, Optional[DataLoader], DataLoader, Counter[int], int]:
    training_roots = resolve_training_roots(args)
    if not training_roots:
        raise ValueError("No training dataset roots configured.")

    train_transform, eval_transform = build_transforms(
        args.filter,
        args.augment_rotation,
        args.augment_scale,
        args.augment_translation,
        args.augment_perspective,
        args.augment_erasing,
        args.augment_noise,
        args.noise_std,
    )

    metadata_maps: dict[Path, dict[str, np.ndarray]] = {}
    metadata_dims: dict[Path, int] = {}
    landmark_dim = 0
    if args.use_landmarks:
        for root in training_roots:
            metadata_file = root / "metadata.csv"
            metadata_map, root_landmark_dim = load_landmark_metadata(metadata_file, root)
            metadata_maps[root] = metadata_map
            metadata_dims[root] = root_landmark_dim
            if landmark_dim == 0 and root_landmark_dim > 0:
                landmark_dim = root_landmark_dim

        for root, root_landmark_dim in metadata_dims.items():
            if root_landmark_dim and landmark_dim and root_landmark_dim != landmark_dim:
                logging.warning(
                    "Landmark dimension for %s is %d (expected %d). Mismatched entries are zero-filled.",
                    root,
                    root_landmark_dim,
                    landmark_dim,
                )
    use_landmarks = args.use_landmarks and landmark_dim > 0

    train_datasets: list[Dataset] = []
    for root in training_roots:
        train_datasets.append(
            _build_split_dataset(
                root=root,
                split="train",
                transform=train_transform,
                use_landmarks=use_landmarks,
                metadata_map=metadata_maps.get(root, {}),
                landmark_dim=landmark_dim,
                args=args,
                train_mode=True,
            )
        )

    train_dataset: Dataset
    if len(train_datasets) == 1:
        train_dataset = train_datasets[0]
    else:
        train_dataset = ConcatDataset(train_datasets)

    if args.sampling == "undersample":
        balanced_indices = _balanced_indices(train_dataset)
        if not balanced_indices:
            raise ValueError("Unable to create undersampled subset: no class targets found.")
        sampled: Dataset | Subset = Subset(train_dataset, balanced_indices)
    else:
        sampled = train_dataset

    sampled_counts = _count_targets(sampled)
    pin_memory = device.type == "cuda"

    val_loader: Optional[DataLoader] = None
    if args.val_data_root is None:
        train_subset, val_subset = _split_dataset(sampled, args.val_split)
        train_loader = DataLoader(
            dataset=train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
        )
        if val_subset is not None:
            val_loader = DataLoader(
                dataset=val_subset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=pin_memory,
            )
    else:
        val_root = args.val_data_root
        val_split_root = val_root / args.val_data_split
        if not val_split_root.exists():
            raise FileNotFoundError(
                f"Validation split folder not found: {val_split_root}"
            )
        if args.val_split > 0:
            logging.info(
                "External validation set is active (%s). Ignoring --val-split=%.2f.",
                val_split_root,
                args.val_split,
            )

        train_loader = DataLoader(
            dataset=sampled,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
        )

        if use_landmarks:
            val_metadata_file = val_root / "metadata.csv"
            val_metadata_map: dict[str, np.ndarray] = {}
            if val_metadata_file.exists():
                val_metadata_map, val_landmark_dim = load_landmark_metadata(
                    val_metadata_file, val_root
                )
                if val_landmark_dim and val_landmark_dim != landmark_dim:
                    logging.warning(
                        "Validation landmark dimension (%d) differs from training (%d). "
                        "Mismatched entries are zero-filled.",
                        val_landmark_dim,
                        landmark_dim,
                    )
            else:
                logging.warning(
                    "Validation metadata %s not found; validation landmarks are zero-filled.",
                    val_metadata_file,
                )

            val_dataset = LandmarkAwareEmotionFolder(
                root=val_split_root,
                transform=None,
                metadata_map=val_metadata_map,
                metadata_root=val_root,
                landmark_dim=landmark_dim,
                filter_name=args.filter,
                augment_rotation=False,
                augment_scale=False,
                augment_translation=False,
                augment_perspective=False,
                augment_erasing=False,
                augment_noise=False,
                noise_std=args.noise_std,
                train_mode=False,
            )
        else:
            val_dataset = EmotionFolderWithPaths(
                root=val_split_root,
                transform=eval_transform,
            )

        filter_classes(val_dataset)
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )

    test_datasets: list[Dataset] = []
    for root in training_roots:
        test_datasets.append(
            _build_split_dataset(
                root=root,
                split="test",
                transform=eval_transform,
                use_landmarks=use_landmarks,
                metadata_map=metadata_maps.get(root, {}),
                landmark_dim=landmark_dim,
                args=args,
                train_mode=False,
            )
        )

    if len(test_datasets) == 1:
        test_dataset = test_datasets[0]
    else:
        test_dataset = ConcatDataset(test_datasets)
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
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_PRESETS.keys()),
        default=None,
        help="Optional dataset presets to concatenate (e.g. --datasets fer2013 affectnet).",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT, help="Path to prepared emotion dataset root")
    parser.add_argument(
        "--extra-data-roots",
        type=Path,
        nargs="*",
        default=[],
        help="Optional extra dataset roots to concatenate with --data-root (ignored when --datasets is used).",
    )
    parser.add_argument(
        "--val-data-root",
        type=Path,
        default=None,
        help="Optional separate dataset root used only for validation (contains split folders like test/).",
    )
    parser.add_argument(
        "--val-data-split",
        type=str,
        default="test",
        help="Split name inside --val-data-root used as validation set (default: test).",
    )
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
    parser.add_argument(
        "--augment-perspective",
        action="store_true",
        help="Add random perspective distortion and small affine shears.",
    )
    parser.add_argument(
        "--augment-erasing",
        action="store_true",
        help="Add RandomErasing (cutout-style occlusion) on tensor images.",
    )
    parser.add_argument("--augment-noise", action="store_true", help="Add Gaussian noise to tensors.")
    parser.add_argument("--noise-std", type=float, default=0.02, help="Standard deviation for Gaussian noise augmentation.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of train set held out for validation.")
    parser.add_argument("--loss", choices=["ce", "mse"], default="ce", help="Loss function used during training.")
    parser.add_argument("--use-landmarks", action="store_true", help="Augment batches with metadata landmarks.")
    parser.add_argument("--use-gnn", action="store_true", help="Enable the relational Landmark GNN branch.")
    parser.add_argument("--gnn-hidden-dim", type=int, default=64, help="Hidden dimensionality for the GNN branch output.")
    parser.add_argument("--gnn-steps", type=int, default=2, help="Message passing rounds for the GNN branch.")
    parser.add_argument("--use-sobel-branch", action="store_true", help="Enable a second feature branch with Sobel gradients.")
    parser.add_argument("--sobel-branch-dim", type=int, default=32, help="Feature dimensionality produced by the Sobel branch.")
    parser.add_argument("--device", type=str, default=None, help="Override training device (cuda/cpu/mps).")
    parser.add_argument(
        "--width-multiplier",
        type=float,
        default=1.0,
        help="Scale factor (<1 for smaller models, >1 for wider).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.35,
        help="Dropout probability applied before the classifier head (0.0 to 1.0).",
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
    training_roots = resolve_training_roots(args)

    augmentation_flags = []
    if args.augment_rotation:
        augmentation_flags.append("rotation")
    if args.augment_scale:
        augmentation_flags.append("scale")
    if args.augment_translation:
        augmentation_flags.append("translation")
    if args.augment_perspective:
        augmentation_flags.append("perspective+skew")
    if args.augment_erasing:
        augmentation_flags.append("random-erasing")
    if args.augment_noise:
        augmentation_flags.append(f"noise(std={args.noise_std})")
    aug_text = "none" if not augmentation_flags else ", ".join(augmentation_flags)
    logging.info(
        "Config: filter=%s, loss=%s, scheduler=%s%s, batch=%d, width=%.2f, dropout=%.2f, val_split=%.2f",
        args.filter,
        args.loss,
        args.scheduler,
        f" (plateau patience={args.plateau_patience} factor={args.plateau_factor})"
        if args.scheduler == "plateau"
        else "",
        args.batch_size,
        args.width_multiplier,
        args.dropout,
        args.val_split,
    )
    logging.info("Augmentations: %s", aug_text)
    logging.info(
        "Landmarks: %s metadata=%s",
        "enabled" if args.use_landmarks else "disabled",
        "<dataset-root>/metadata.csv",
    )
    logging.info(
        "GNN branch request: %s (hidden=%d steps=%d)",
        "on" if args.use_gnn else "off",
        args.gnn_hidden_dim,
        args.gnn_steps,
    )
    logging.info(
        "Sobel branch request: %s (dim=%d)",
        "on" if args.use_sobel_branch else "off",
        args.sobel_branch_dim,
    )

    device = torch.device(args.device) if args.device else DEFAULT_DEVICE
    torch.manual_seed(RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(RANDOM_SEED)

    roots_text = ", ".join(str(root) for root in training_roots)
    logging.info(
        "Loading data from %s (filter=%s, sampling=%s, width=%.2f, landmarks=%s)",
        roots_text,
        args.filter,
        args.sampling,
        args.width_multiplier,
        args.use_landmarks,
    )
    if len(training_roots) > 1 and any("fer2013" in root.as_posix().lower() for root in training_roots):
        logging.info("FER2013 inputs (48x48) are upscaled to 64x64 in the preprocessing pipeline.")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader, class_counts, landmark_dim = build_dataloaders(args, device)
    gnn_requested = args.use_gnn
    landmarks_requested = args.use_landmarks
    landmark_features_available = landmark_dim > 0
    gnn_active = gnn_requested and landmarks_requested and landmark_features_available
    if gnn_requested and not landmarks_requested:
        logging.warning("GNN branch requested but --use-landmarks is disabled; enable landmarks to activate the GNN.")
    elif gnn_requested and not landmark_features_available:
        logging.warning("GNN branch requested but landmark metadata could not be loaded; branch stays inactive.")
    status = "active" if gnn_active else "inactive"
    logging.info(
        "GNN branch status: requested=%s landmarks=%s dim=%d -> %s (hidden=%d steps=%d)",
        gnn_requested,
        landmarks_requested,
        landmark_dim,
        status,
        args.gnn_hidden_dim,
        args.gnn_steps,
    )
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
        dropout=args.dropout,
        landmark_dim=landmark_dim if args.use_landmarks else 0,
        use_gnn=args.use_gnn,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_message_steps=args.gnn_steps,
        use_sobel_branch=args.use_sobel_branch,
        sobel_branch_dim=args.sobel_branch_dim,
    ).to(device)
    log_model_summary(model)
    logging.info("Model architecture:\\n%s", model)
    if model.landmark_attention is not None:
        logging.info("Landmark attention shading: heatmap_channels=%d branch_feat_dim=%d",
            model.landmark_attention.heatmap_channels,
            model.landmark_attention.feature_dim,
        )
    if model.gnn_branch is not None:
        logging.info("Landmark GNN branch: hidden_dim=%d message_steps=%d",
            model.gnn_branch.hidden_dim,
            model.gnn_branch.message_steps,
        )
    if model.sobel_branch is not None:
        logging.info("Sobel branch: feature_dim=%d", model.sobel_branch.feature_dim)

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
                torch.save(model.state_dict(), BEST_CHECKPOINT)
                logging.info("Saved improved model to %s (val acc %.2f%%)", checkpoint, val_acc)
                logging.info("Updated best checkpoint at %s", BEST_CHECKPOINT)

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
