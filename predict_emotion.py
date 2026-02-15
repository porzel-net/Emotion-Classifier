#!/usr/bin/env python3
"""Predict emotion from webcam or video file and overlay results."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

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
                    with torch.no_grad():
                        logits = model(tensor)
                        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                        pred_idx = int(np.argmax(probs))
                        label = EMOTION_LABELS[pred_idx]
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
