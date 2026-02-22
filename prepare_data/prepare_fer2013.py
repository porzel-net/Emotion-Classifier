import argparse
import csv
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import face_alignment
import numpy as np
from helpers.neconet_helpers import apply_sobel


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

EXCLUDED_EMOTIONS = {"neutral"}


def format_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    total_seconds = int(seconds)
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass
class ProgressTracker:
    total: int
    seen: int = 0
    saved: int = 0
    skipped: int = 0
    start_time: float = field(default_factory=time.monotonic)
    last_log_time: float = field(default_factory=lambda: 0.0)
    log_interval_sec: float = 5.0

    def update(self, saved: bool) -> None:
        self.seen += 1
        if saved:
            self.saved += 1
        else:
            self.skipped += 1
        self.log()

    def log(self, force: bool = False) -> None:
        if self.total == 0:
            if force:
                logging.info("Progress: 100.0%% (0/0) | saved=0 skipped=0 | elapsed=00:00 | ETA=00:00")
            return

        now = time.monotonic()
        if not force and self.seen < self.total and (now - self.last_log_time) < self.log_interval_sec:
            return

        elapsed = now - self.start_time
        rate = self.seen / elapsed if elapsed > 0 else 0.0
        remaining = self.total - self.seen
        eta = (remaining / rate) if rate > 0 else 0.0
        percent = (self.seen / self.total) * 100.0

        logging.info(
            "Progress: %.1f%% (%d/%d) | saved=%d skipped=%d | elapsed=%s | ETA=%s",
            percent,
            self.seen,
            self.total,
            self.saved,
            self.skipped,
            format_duration(elapsed),
            format_duration(eta),
        )
        self.last_log_time = now


def detect_landmarks(model, image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preds = model.get_landmarks(rgb)
    return preds[0] if preds else None


def scale_landmarks(landmarks, source_shape, target_shape):
    ratio_y = target_shape[0] / source_shape[0]
    ratio_x = target_shape[1] / source_shape[1]
    scaled = landmarks.copy()
    scaled[:, 0] *= ratio_x
    scaled[:, 1] *= ratio_y
    return scaled


def normalize_landmarks(landmarks, target_shape):
    normalized = landmarks.copy()
    normalized[:, 0] /= target_shape[1]
    normalized[:, 1] /= target_shape[0]
    return normalized


def process_split(model, split_path, split_name, target_root, target_size, progress):
    metadata = []
    for emotion_dir in sorted(split_path.iterdir()):
        if not emotion_dir.is_dir():
            continue
        if emotion_dir.name.lower() in EXCLUDED_EMOTIONS:
            logging.info("Skipping excluded class %s/%s", split_name, emotion_dir.name)
            continue

        dest_dir = target_root / split_name / emotion_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)

        logging.info("Processing %s/%s", split_name, emotion_dir.name)
        for image_path in sorted(emotion_dir.glob("*")):
            if not image_path.is_file():
                continue

            frame = cv2.imread(str(image_path))
            if frame is None:
                logging.warning("Skipped unreadable image %s", image_path)
                progress.update(saved=False)
                continue

            landmarks = detect_landmarks(model, frame)
            if landmarks is None:
                logging.warning("No landmarks for %s", image_path)
                progress.update(saved=False)
                continue

            logging.debug(
                "Detected %d landmarks for %s",
                len(landmarks),
                image_path,
            )

            resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            sobel_frame = apply_sobel(resized)

            scaled_landmarks = scale_landmarks(
                np.array(landmarks), frame.shape[:2], target_size
            )
            normalized_landmarks = normalize_landmarks(
                scaled_landmarks, target_size
            )

            rel_output = (
                dest_dir / image_path.name
            )
            write_ok = cv2.imwrite(str(rel_output), sobel_frame)
            if not write_ok:
                logging.warning("Failed to write output image %s", rel_output)
                progress.update(saved=False)
                continue

            metadata.append(
                {
                    "split": split_name,
                    "emotion": emotion_dir.name,
                    "image_path": str(rel_output.relative_to(target_root)),
                    "source_path": str(image_path.relative_to(split_path.parent)),
                    "landmarks": json.dumps(
                        normalized_landmarks.tolist(), ensure_ascii=False
                    ),
                }
            )
            progress.update(saved=True)
    return metadata


def build_dataset(args):
    input_root = Path(args.source)
    target_root = Path(args.target)

    if not input_root.exists():
        raise FileNotFoundError(f"{input_root} does not exist")

    if target_root.exists():
        logging.info("Removing existing output at %s", target_root)
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    model = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device=args.device,
    )

    target_size = (args.resize, args.resize)
    metadata = []
    split_paths: list[tuple[str, Path]] = []
    total_candidates = 0

    for split_name in ["train", "test"]:
        split_path = input_root / split_name
        if not split_path.exists():
            logging.warning("Skipping missing split %s", split_path)
            continue
        split_paths.append((split_name, split_path))
        for emotion_dir in split_path.iterdir():
            if not emotion_dir.is_dir():
                continue
            if emotion_dir.name.lower() in EXCLUDED_EMOTIONS:
                continue
            total_candidates += sum(1 for image_path in emotion_dir.glob("*") if image_path.is_file())

    progress = ProgressTracker(total=total_candidates)
    logging.info("Found %d input images for processing", total_candidates)

    for split_name, split_path in split_paths:
        metadata.extend(
            process_split(
                model,
                split_path,
                split_name,
                target_root,
                target_size,
                progress,
            )
        )

    progress.log(force=True)

    metadata_path = target_root / "metadata.csv"
    with open(metadata_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "emotion", "image_path", "source_path", "landmarks"],
        )
        writer.writeheader()
        writer.writerows(metadata)

    logging.info("Finished. Dataset ready at %s", target_root)
    return metadata_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare landmark-aware emotion classifier dataset"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/fer2013"),
        help="Source FER dataset root (must contain train/ and test/)",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("data/fer2013-prepared"),
        help="Destination root for the processed dataset",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=48,
        help="Side length to resize the final image to (keep original 48x48)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for face_alignment (cpu/mps/cuda)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
