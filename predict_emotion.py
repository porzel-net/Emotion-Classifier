#!/usr/bin/env python3
"""Predict emotion from webcam or video file and overlay results."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import cv2
import face_alignment
import numpy as np
import torch
import warnings
from PIL import Image

from neconet_helpers import EMOTION_LABELS, build_model, build_transform, get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

INPUT_TRANSFORM = build_transform()
EXPLAIN_METHODS = ("saliency", "activations", "occlusion", "cam", "gradcam")


def _normalize_heatmap(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().clone()
    tensor = tensor - tensor.min()
    max_val = tensor.max()
    if max_val > 0:
        tensor = tensor / max_val
    return tensor.clamp(0, 1)


def _tensor_to_heatmap(tensor: torch.Tensor) -> np.ndarray:
    return _normalize_heatmap(tensor).cpu().numpy()


def _overlay_heatmap(frame: np.ndarray, box: tuple[tuple[int, int], tuple[int, int]], heatmap: np.ndarray) -> None:
    (x1, y1), (x2, y2) = box
    if x2 <= x1 or y2 <= y1:
        return
    overlay = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.resize(overlay, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    roi = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.5, overlay, 0.5, 0)


class LayerCapture:
    """Hook into a module to capture activations (and gradients when available)."""

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._forward = module.register_forward_hook(self._save_activation)
        self._backward = module.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: torch.nn.Module, _input, output: torch.Tensor) -> None:
        self.activations = output

    def _save_gradient(self, module: torch.nn.Module, _grad_input, grad_output) -> None:
        grad = grad_output[0]
        if grad is not None:
            self.gradients = grad

    def reset(self) -> None:
        self.activations = None
        self.gradients = None

    def close(self) -> None:
        self._forward.remove()
        self._backward.remove()


def _compute_saliency_heatmap(model: torch.nn.Module, logits: torch.Tensor, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    score = logits[0, class_idx]
    score.backward()
    grads = tensor.grad.abs().squeeze(0).max(dim=0).values
    tensor.grad.zero_()
    return _tensor_to_heatmap(grads)


def _compute_activation_heatmap(capture: LayerCapture) -> np.ndarray:
    if capture.activations is None:
        return np.zeros((1, 1))
    activation = capture.activations.detach().squeeze(0)
    heatmap = activation.abs().mean(dim=0)
    return _tensor_to_heatmap(heatmap)


def _compute_cam_heatmap(model: torch.nn.Module, capture: LayerCapture, class_idx: int) -> np.ndarray:
    if capture.activations is None:
        return np.zeros((1, 1))
    weights = model.fc.weight[class_idx].detach()
    activation = capture.activations.detach().squeeze(0)
    cam = (weights[:, None, None] * activation).sum(dim=0)
    return _tensor_to_heatmap(torch.relu(cam))


def _compute_gradcam_heatmap(model: torch.nn.Module, logits: torch.Tensor, capture: LayerCapture, class_idx: int) -> np.ndarray:
    if capture.activations is None:
        return np.zeros((1, 1))
    model.zero_grad(set_to_none=True)
    score = logits[0, class_idx]
    score.backward()
    if capture.gradients is None:
        return np.zeros((1, 1))
    gradients = capture.gradients.detach()
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * capture.activations).sum(dim=1).squeeze(0)
    return _tensor_to_heatmap(torch.relu(cam))


def _compute_occlusion_heatmap(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    class_idx: int,
    base_prob: float,
    patch_size: int,
    stride: int,
) -> np.ndarray:
    h, w = tensor.shape[-2:]
    device = tensor.device
    heatmap = torch.zeros((h, w), device=device)
    baseline = tensor.mean()
    with torch.no_grad():
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                occluded = tensor.clone()
                occluded[..., y:y_end, x:x_end] = baseline
                probs = torch.softmax(model(occluded), dim=1)
                drop = base_prob - probs[0, class_idx].item()
                if drop < 0:
                    drop = 0
                heatmap[y:y_end, x:x_end] = drop
    return _tensor_to_heatmap(heatmap)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict emotions from webcam or video.")
    parser.add_argument(
        "--weights",
        "--weights-dir",
        dest="weights",
        type=Path,
        default=Path("Neconet_Weights3.pth"),
        help="Path to .pth checkpoint or weights directory.",
    )
    parser.add_argument("--device", default=None, help="Force torch device such as cpu/mps/cuda.")
    parser.add_argument("--webcam", action="store_true", help="Use the webcam as input.")
    parser.add_argument("--video", type=Path, help="Path to a video file to process.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output video path (defaults to INPUT_annotated.mp4 when --video is used).",
    )
    parser.add_argument("--scale", type=float, default=0.4, help="Downscale factor for landmark detection.")
    parser.add_argument("--skip-frames", type=int, default=2, help="Skip frames between landmark detections.")
    parser.add_argument("--show", action="store_true", help="Show the processed frames during video processing.")
    parser.add_argument(
        "--explain",
        choices=EXPLAIN_METHODS,
        help="Overlay an explanation heatmap for the detected face (saliency, activations, occlusion, CAM, GradCAM).",
    )
    parser.add_argument(
        "--occlusion-patch",
        type=int,
        default=8,
        help="Patch size (in transformed pixels) for occlusion sensitivity.",
    )
    parser.add_argument(
        "--occlusion-stride",
        type=int,
        default=4,
        help="Stride (in transformed pixels) between occlusion patches.",
    )
    return parser.parse_args()


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    if args.webcam:
        cap = cv2.VideoCapture(0)
    elif args.video:
        cap = cv2.VideoCapture(str(args.video))
    else:
        raise ValueError("Either --webcam or --video must be provided.")
    if not cap.isOpened():
        raise RuntimeError("Unable to open capture device")
    return cap


def build_video_writer(cap: cv2.VideoCapture, output: Path) -> cv2.VideoWriter:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(output), fourcc, fps, (width, height))


def draw_overlay(frame: np.ndarray, probs: np.ndarray, label: str, crop_box: tuple[tuple[int, int], tuple[int, int]]):
    start_x = frame.shape[1] - 220
    start_y = 20
    cv2.rectangle(frame, crop_box[0], crop_box[1], (255, 0, 0), 2)
    cv2.putText(
        frame,
        label,
        (crop_box[0][0], max(crop_box[0][1] - 10, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for idx, name in enumerate(EMOTION_LABELS):
        text = f"{name[:3]}: {probs[idx] * 100:4.1f}%"
        cv2.putText(
            frame,
            text,
            (start_x, start_y + idx * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    display_window = args.webcam or args.show
    if not (args.webcam or args.video):
        LOGGER.error("Provide --webcam or --video.")
        return

    device = get_device(args.device)
    LOGGER.info("Using device %s", device)
    model = build_model(args.weights, device, logger=LOGGER)
    fa_device = "cuda" if device.type == "cuda" else "cpu"
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=fa_device)
    capture: Optional[LayerCapture] = None
    if args.explain in {"activations", "cam", "gradcam"}:
        capture = LayerCapture(model.layer4)

    cap = open_capture(args)
    writer = None
    if args.video:
        output = args.output or args.video.with_name(f"{args.video.stem}_annotated.mp4")
        writer = build_video_writer(cap, output)
        LOGGER.info("Writing annotated video to %s", output)

    frame_idx = 0
    current_box: tuple[tuple[int, int], tuple[int, int]] | None = None
    last_landmarks: np.ndarray | None = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            detect_frame = cv2.resize(frame, (0, 0), fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
            gray_detect = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)

            if frame_idx % args.skip_frames == 0:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="No faces were detected\\.")
                    faces = fa.get_landmarks(gray_detect)
                if faces:
                    landmarks = faces[0]
                    last_landmarks = landmarks
                    min_x, min_y = np.min(landmarks[:, 0]) / args.scale, np.min(landmarks[:, 1]) / args.scale
                    max_x, max_y = np.max(landmarks[:, 0]) / args.scale, np.max(landmarks[:, 1]) / args.scale
                    x1, y1 = int(max(min_x, 0)), int(max(min_y, 0))
                    x2, y2 = int(min(max_x, frame.shape[1])), int(min(max_y, frame.shape[0]))
                    if x2 > x1 and y2 > y1:
                        current_box = ((x1, y1), (x2, y2))

            if last_landmarks is not None:
                for (x, y) in last_landmarks:
                    cv2.circle(frame, (int(x / args.scale), int(y / args.scale)), 2, (0, 255, 0), -1)

            if current_box:
                (x1, y1), (x2, y2) = current_box
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    tensor = INPUT_TRANSFORM(pil_crop).unsqueeze(0).to(device)
                    grad_enabled = args.explain in {"saliency", "gradcam"}
                    if grad_enabled:
                        tensor.requires_grad_(True)
                    with torch.set_grad_enabled(grad_enabled):
                        logits = model(tensor)
                        softmax = torch.softmax(logits, dim=1)
                    probs = softmax.squeeze(0).detach().cpu().numpy()
                    pred_idx = int(np.argmax(probs))
                    label = EMOTION_LABELS[pred_idx]
                    if args.explain:
                        heatmap: Optional[np.ndarray] = None
                        if args.explain == "saliency":
                            heatmap = _compute_saliency_heatmap(model, logits, tensor, pred_idx)
                        elif args.explain == "activations" and capture:
                            heatmap = _compute_activation_heatmap(capture)
                        elif args.explain == "cam" and capture:
                            heatmap = _compute_cam_heatmap(model, capture, pred_idx)
                        elif args.explain == "gradcam" and capture:
                            heatmap = _compute_gradcam_heatmap(model, logits, capture, pred_idx)
                            capture.reset()
                        elif args.explain == "occlusion":
                            heatmap = _compute_occlusion_heatmap(
                                model,
                                tensor,
                                pred_idx,
                                softmax[0, pred_idx].item(),
                                args.occlusion_patch,
                                args.occlusion_stride,
                            )
                        if heatmap is not None:
                            _overlay_heatmap(frame, ((x1, y1), (x2, y2)), heatmap)
                    draw_overlay(frame, probs, label, ((x1, y1), (x2, y2)))

            if writer:
                writer.write(frame)
            if display_window:
                cv2.imshow("Emotion Predictor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        if display_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
