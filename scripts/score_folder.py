#!/usr/bin/env python3
"""Score images from a folder and write class probabilities as CSV."""

from __future__ import annotations

import argparse
import csv
import logging
import warnings
from pathlib import Path
from typing import Any

import cv2
import face_alignment
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from helpers.model_architecture import Block as LegacyBlock
from helpers.model_architecture import ResNet as LegacyResNet
from helpers.neconet_helpers import (
    EMOTION_LABELS,
    apply_laplacian_to_pil,
    apply_roberts_to_pil,
    apply_sobel_to_pil,
    get_device,
    load_state_dict,
)
from train.train_emotion_classifier import Block as EvalBlock
from train.train_emotion_classifier import EmotionResNet

LOGGER = logging.getLogger(__name__)
IMAGE_SIZE = 64
CSV_COLUMNS = ["filepath", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]
CSV_ORDER = ["Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear"]
CLASS_ORDER = ["angry", "disgusted", "fearful", "happy", "sad", "surprised"]
DEFAULT_MODEL_WEIGHTS = Path("models/emotion-classifier-best.pth")

FILTER_TRANSFORMS = {
    "sobel": transforms.Lambda(apply_sobel_to_pil),
    "roberts": transforms.Lambda(apply_roberts_to_pil),
    "laplacian": transforms.Lambda(apply_laplacian_to_pil),
    "none": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score each image in a folder and emit CSV probabilities."
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
        help="Checkpoint path (default: models/emotion-classifier-best.pth).",
    )
    parser.add_argument("--device", default=None, help="Torch device to use (cpu/mps/cuda).")
    parser.add_argument(
        "--filter",
        choices=list(FILTER_TRANSFORMS.keys()),
        default="sobel",
        help="Preprocessing filter used during training.",
    )
    parser.add_argument(
        "--width-multiplier",
        type=float,
        default=None,
        help="Optional width multiplier override (auto-inferred from checkpoint by default).",
    )
    parser.add_argument(
        "--use-landmarks",
        action="store_true",
        help="Kept for compatibility. Landmark branch is inferred from checkpoint.",
    )
    parser.add_argument(
        "--use-gnn",
        action="store_true",
        help="Kept for compatibility. GNN branch is inferred from checkpoint.",
    )
    parser.add_argument(
        "--gnn-hidden-dim",
        type=int,
        default=None,
        help="Optional GNN hidden dim override (auto-inferred from checkpoint by default).",
    )
    parser.add_argument("--gnn-steps", type=int, default=2, help="GNN message steps (must match training).")
    parser.add_argument(
        "--face-device",
        default=None,
        help="Device for face landmark extraction (cpu/cuda). Defaults to cuda when torch device is cuda, else cpu.",
    )
    parser.add_argument(
        "--crop-pad",
        type=float,
        default=0.08,
        help="Padding ratio around landmark box before cropping.",
    )
    parser.add_argument(
        "--no-face-analysis",
        action="store_true",
        help="Disable face landmark detection and face cropping before inference.",
    )
    return parser.parse_args()


def build_eval_transform(filter_name: str) -> transforms.Compose:
    ops: list[transforms.Transform] = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
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


def _normalize_state_dict(raw_state: Any) -> dict[str, torch.Tensor]:
    state = raw_state
    if isinstance(state, dict):
        if isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        elif isinstance(state.get("model_state_dict"), dict):
            state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state)!r}")
    if state and all(isinstance(key, str) and key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def _layer4_channels(state: dict[str, torch.Tensor]) -> int:
    for key, tensor in state.items():
        if key.startswith("layer4.") and key.endswith(".conv2.weight") and tensor.ndim == 4:
            return int(tensor.shape[0])
    tensor = state.get("layer4.0.conv1.weight")
    if tensor is None or tensor.ndim != 4:
        raise KeyError("Unable to infer layer4 channel count from checkpoint.")
    return int(tensor.shape[0])


def _is_legacy_resnet_state(state: dict[str, torch.Tensor]) -> bool:
    if any(key.startswith(("landmark_attention.", "gnn_branch.", "sobel_branch.")) for key in state):
        return False
    return "conv1.bias" in state


def _infer_emotion_resnet_config(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    conv1 = state.get("conv1.weight")
    fc_weight = state.get("fc.weight")
    if conv1 is None or fc_weight is None:
        raise KeyError("Checkpoint is missing required keys (conv1.weight/fc.weight).")

    width_multiplier = max(0.25, float(conv1.shape[0]) / 64.0)
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

    return {
        "width_multiplier": width_multiplier,
        "num_classes": num_classes,
        "landmark_dim": landmark_dim,
        "use_gnn": use_gnn,
        "gnn_hidden_dim": gnn_hidden_dim,
        "use_sobel_branch": use_sobel_branch,
        "sobel_branch_dim": sobel_branch_dim,
    }


def build_evaluation_model(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, int]:
    state = _normalize_state_dict(load_state_dict(args.weights, device))

    if _is_legacy_resnet_state(state):
        model: torch.nn.Module = LegacyResNet(LegacyBlock, (2, 2, 2, 2), len(CLASS_ORDER))
        model.to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        model._expects_landmarks = False  # type: ignore[attr-defined]
        model._model_info = {"architecture": "ResNet", "width_multiplier": 1.0}  # type: ignore[attr-defined]
        return model, 0

    cfg = _infer_emotion_resnet_config(state)
    if args.width_multiplier is not None and abs(args.width_multiplier - cfg["width_multiplier"]) > 1e-6:
        LOGGER.warning(
            "Ignoring --width-multiplier=%.3f because checkpoint requires %.3f.",
            args.width_multiplier,
            cfg["width_multiplier"],
        )
    if args.gnn_hidden_dim is not None and args.gnn_hidden_dim != cfg["gnn_hidden_dim"]:
        LOGGER.warning(
            "Ignoring --gnn-hidden-dim=%d because checkpoint requires %d.",
            args.gnn_hidden_dim,
            cfg["gnn_hidden_dim"],
        )
    if args.use_landmarks and cfg["landmark_dim"] == 0:
        LOGGER.warning("--use-landmarks was set, but checkpoint has no landmark branch; flag is ignored.")
    if args.use_gnn and not cfg["use_gnn"]:
        LOGGER.warning("--use-gnn was set, but checkpoint has no GNN branch; flag is ignored.")

    landmark_dim = int(cfg["landmark_dim"])
    model = EmotionResNet(
        EvalBlock,
        layers=(2, 2, 2, 2),
        num_classes=int(cfg["num_classes"]),
        width_multiplier=float(cfg["width_multiplier"]),
        landmark_dim=landmark_dim,
        use_gnn=bool(cfg["use_gnn"]),
        gnn_hidden_dim=int(cfg["gnn_hidden_dim"]),
        gnn_message_steps=args.gnn_steps,
        use_sobel_branch=bool(cfg["use_sobel_branch"]),
        sobel_branch_dim=int(cfg["sobel_branch_dim"]),
    )
    model.to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    model._expects_landmarks = landmark_dim > 0  # type: ignore[attr-defined]
    model._model_info = {  # type: ignore[attr-defined]
        "architecture": "EmotionResNet",
        "width_multiplier": float(cfg["width_multiplier"]),
        "landmarks": landmark_dim > 0,
        "gnn": bool(cfg["use_gnn"]),
        "sobel_branch": bool(cfg["use_sobel_branch"]),
    }
    return model, landmark_dim


def build_face_aligner(device: torch.device, face_device_choice: str | None):
    face_device = face_device_choice or ("cuda" if device.type == "cuda" else "cpu")
    return face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device=face_device,
    ), face_device


def detect_primary_landmarks(aligner, frame_bgr: np.ndarray) -> np.ndarray | None:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No faces were detected\\.")
        preds = aligner.get_landmarks(rgb)
    if not preds:
        return None

    def _area(points: np.ndarray) -> float:
        min_x = float(np.min(points[:, 0]))
        max_x = float(np.max(points[:, 0]))
        min_y = float(np.min(points[:, 1]))
        max_y = float(np.max(points[:, 1]))
        return max(0.0, (max_x - min_x) * (max_y - min_y))

    best = max((np.asarray(p, dtype=np.float32) for p in preds), key=_area)
    return best


def crop_from_landmarks(
    frame: np.ndarray,
    landmarks: np.ndarray,
    pad_ratio: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = frame.shape[:2]
    min_x = float(np.min(landmarks[:, 0]))
    max_x = float(np.max(landmarks[:, 0]))
    min_y = float(np.min(landmarks[:, 1]))
    max_y = float(np.max(landmarks[:, 1]))

    box_w = max_x - min_x
    box_h = max_y - min_y
    if box_w <= 1.0 or box_h <= 1.0:
        return None

    pad_x = box_w * max(0.0, pad_ratio)
    pad_y = box_h * max(0.0, pad_ratio)

    x1 = max(0, int(np.floor(min_x - pad_x)))
    y1 = max(0, int(np.floor(min_y - pad_y)))
    x2 = min(w, int(np.ceil(max_x + pad_x)))
    y2 = min(h, int(np.ceil(max_y + pad_y)))
    if x2 <= x1 or y2 <= y1:
        return None

    cropped = frame[y1:y2, x1:x2]
    shifted = landmarks.copy()
    shifted[:, 0] -= x1
    shifted[:, 1] -= y1
    return cropped, shifted


def landmarks_to_vector(
    landmarks: np.ndarray,
    source_shape: tuple[int, int],
    target_size: int,
    expected_dim: int,
) -> tuple[np.ndarray, bool]:
    if expected_dim <= 0:
        return np.zeros((0,), dtype=np.float32), False

    src_h, src_w = source_shape
    if src_h <= 0 or src_w <= 0:
        return np.zeros((expected_dim,), dtype=np.float32), True

    resized = landmarks.copy().astype(np.float32)
    resized[:, 0] *= target_size / float(src_w)
    resized[:, 1] *= target_size / float(src_h)
    normalized = resized
    normalized[:, 0] /= float(target_size)
    normalized[:, 1] /= float(target_size)
    normalized = np.clip(normalized, 0.0, 1.0)
    vec = normalized.reshape(-1).astype(np.float32)

    if vec.size == expected_dim:
        return vec, False

    out = np.zeros((expected_dim,), dtype=np.float32)
    copy_len = min(expected_dim, vec.size)
    if copy_len > 0:
        out[:copy_len] = vec[:copy_len]
    return out, True


def prepare_face_sample(
    path: Path,
    aligner,
    crop_pad: float,
    landmark_dim: int,
) -> tuple[Image.Image, np.ndarray, bool, bool, bool]:
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Unreadable image: {path}")

    landmarks = detect_primary_landmarks(aligner, frame)
    found_landmarks = landmarks is not None
    used_crop = False
    dim_adjusted = False
    landmark_vector = np.zeros((landmark_dim,), dtype=np.float32)

    face_region = frame
    working_landmarks: np.ndarray | None = None
    if landmarks is not None:
        cropped = crop_from_landmarks(frame, landmarks, pad_ratio=crop_pad)
        if cropped is not None:
            face_region, shifted = cropped
            working_landmarks = shifted
            used_crop = True
        else:
            working_landmarks = landmarks

    if landmark_dim > 0 and working_landmarks is not None:
        landmark_vector, dim_adjusted = landmarks_to_vector(
            working_landmarks,
            source_shape=face_region.shape[:2],
            target_size=IMAGE_SIZE,
            expected_dim=landmark_dim,
        )

    pil_image = Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
    return pil_image, landmark_vector, found_landmarks, used_crop, dim_adjusted


def reorder_probs(probs: list[float]) -> list[float]:
    order_map = {name: idx for idx, name in enumerate(EMOTION_LABELS)}
    return [probs[order_map[label]] for label in CSV_ORDER]


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
    expects_landmarks = bool(getattr(model, "_expects_landmarks", False))
    model_info = getattr(model, "_model_info", {"architecture": type(model).__name__, "width_multiplier": 1.0})
    LOGGER.info(
        "Loaded evaluation model (arch=%s, filter=%s, width=%.2f, landmarks=%s, gnn=%s, sobel_branch=%s, landmark_dim=%d)",
        model_info.get("architecture", type(model).__name__),
        args.filter,
        float(model_info.get("width_multiplier", 1.0)),
        bool(model_info.get("landmarks", expects_landmarks)),
        bool(model_info.get("gnn", False)),
        bool(model_info.get("sobel_branch", False)),
        landmark_dim,
    )

    aligner = None
    face_landmark_device = "disabled"
    if not args.no_face_analysis:
        aligner, face_landmark_device = build_face_aligner(device, args.face_device)
        LOGGER.info(
            "Face analysis enabled (landmark_device=%s, crop_pad=%.3f).",
            face_landmark_device,
            args.crop_pad,
        )
    else:
        LOGGER.info("Face analysis disabled via --no-face-analysis.")

    image_paths = sorted(p for p in args.images.iterdir() if p.is_file())
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.images}")

    rows: list[tuple[str, list[float]]] = []
    skipped = 0
    found_landmarks = 0
    used_crops = 0
    adjusted_vectors = 0

    for path in image_paths:
        try:
            if aligner is not None:
                pil_image, landmark_vector, found, cropped, adjusted = prepare_face_sample(
                    path,
                    aligner,
                    crop_pad=args.crop_pad,
                    landmark_dim=landmark_dim,
                )
                found_landmarks += int(found)
                used_crops += int(cropped)
                adjusted_vectors += int(adjusted)
            else:
                pil_image = Image.open(path).convert("RGB")
                landmark_vector = np.zeros((landmark_dim,), dtype=np.float32)

            tensor = transform(pil_image).unsqueeze(0).to(device)
            landmarks_tensor = None
            if expects_landmarks:
                landmarks_tensor = torch.from_numpy(landmark_vector).float().unsqueeze(0).to(device)

            with torch.no_grad():
                if expects_landmarks:
                    logits = model(tensor, landmarks_tensor)
                else:
                    logits = model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().tolist()
            rows.append((path.as_posix(), reorder_probs(probs)))
        except Exception as exc:  # pragma: no cover - defensive for large folder runs
            skipped += 1
            LOGGER.warning("Skipping %s: %s", path, exc)

    if aligner is not None:
        LOGGER.info(
            "Face analysis summary: landmarks=%d/%d, crops=%d/%d, adjusted_landmark_vectors=%d",
            found_landmarks,
            len(image_paths),
            used_crops,
            len(image_paths),
            adjusted_vectors,
        )

    LOGGER.info("Scored %d/%d images (skipped=%d).", len(rows), len(image_paths), skipped)
    LOGGER.info("Writing %d rows to %s", len(rows), args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for filepath, probs in rows:
            writer.writerow([filepath] + [f"{score:.2f}" for score in probs])


if __name__ == "__main__":
    main()
