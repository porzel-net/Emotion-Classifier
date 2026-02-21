import argparse
import csv
import json
import logging
import shutil
from pathlib import Path

import cv2
import face_alignment
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# AffectNet YOLO classes -> FER-style folder names
AFFECTNET_TO_FER = {
    0: "angry",
    1: None,  # contempt not present in FER-2013
    2: "disgusted",
    3: "fearful",
    4: "happy",
    5: "neutral",
    6: "sad",
    7: "surprised",
}

# Keep FER-like output with train/test only.
SPLIT_MAP = {
    "train": "train",
    "valid": "test",
    "test": "test",
}


def parse_label_class(label_path: Path) -> int | None:
    if not label_path.exists():
        return None
    line = label_path.read_text(encoding="utf-8").strip().splitlines()
    if not line:
        return None
    parts = line[0].strip().split()
    if not parts:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None


def detect_face_bbox(detector, frame: np.ndarray) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda bb: bb[2] * bb[3])
    return int(x), int(y), int(w), int(h)


def detect_landmarks_in_face(model, frame: np.ndarray, face_bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    x, y, w, h = face_bbox
    face_crop = frame[y : y + h, x : x + w]
    if face_crop.size == 0:
        return None

    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    preds = model.get_landmarks(rgb)
    if not preds:
        return None

    landmarks = np.array(preds[0], dtype=np.float32)
    landmarks[:, 0] += x
    landmarks[:, 1] += y
    return landmarks


def crop_from_landmarks(frame: np.ndarray, landmarks: np.ndarray, pad_ratio: float = 0.08) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = frame.shape[:2]

    min_x = float(np.min(landmarks[:, 0]))
    max_x = float(np.max(landmarks[:, 0]))
    min_y = float(np.min(landmarks[:, 1]))
    max_y = float(np.max(landmarks[:, 1]))

    box_w = max_x - min_x
    box_h = max_y - min_y
    if box_w <= 1 or box_h <= 1:
        return None

    pad_x = box_w * pad_ratio
    pad_y = box_h * pad_ratio

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


def resize_landmarks(landmarks: np.ndarray, src_shape: tuple[int, int], target_size: int) -> np.ndarray:
    src_h, src_w = src_shape
    scale_x = target_size / src_w
    scale_y = target_size / src_h
    resized = landmarks.copy()
    resized[:, 0] *= scale_x
    resized[:, 1] *= scale_y
    return resized


def normalize_landmarks(landmarks: np.ndarray, size: int) -> np.ndarray:
    out = landmarks.copy()
    out[:, 0] /= size
    out[:, 1] /= size
    return out


def unique_output_path(dest_dir: Path, base_name: str) -> Path:
    candidate = dest_dir / f"{base_name}.png"
    if not candidate.exists():
        return candidate
    idx = 1
    while True:
        candidate = dest_dir / f"{base_name}_{idx}.png"
        if not candidate.exists():
            return candidate
        idx += 1


def process_split(
    model,
    detector,
    source_root: Path,
    source_split: str,
    target_root: Path,
    target_size: int,
    max_samples: int | None,
) -> list[dict[str, str]]:
    metadata: list[dict[str, str]] = []

    images_dir = source_root / source_split / "images"
    labels_dir = source_root / source_split / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        logging.warning("Skipping missing split folders for %s", source_split)
        return metadata

    target_split = SPLIT_MAP[source_split]
    processed = 0

    for image_path in sorted(images_dir.glob("*")):
        if max_samples is not None and processed >= max_samples:
            break
        if not image_path.is_file():
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"
        class_id = parse_label_class(label_path)
        if class_id is None:
            continue

        emotion_name = AFFECTNET_TO_FER.get(class_id)
        if emotion_name is None:
            continue

        frame = cv2.imread(str(image_path))
        if frame is None:
            logging.warning("Skipped unreadable image %s", image_path)
            continue

        face_bbox = detect_face_bbox(detector, frame)
        if face_bbox is None:
            logging.debug("No face detected for %s", image_path)
            continue

        landmarks = detect_landmarks_in_face(model, frame, face_bbox)
        if landmarks is None:
            logging.debug("No landmarks detected for %s", image_path)
            continue

        cropped_result = crop_from_landmarks(frame, landmarks)
        if cropped_result is None:
            continue

        cropped, shifted_landmarks = cropped_result
        resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        resized_landmarks = resize_landmarks(shifted_landmarks, cropped.shape[:2], target_size)
        normalized_landmarks = normalize_landmarks(resized_landmarks, target_size)

        dest_dir = target_root / target_split / emotion_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_name_base = f"{source_split}__{image_path.stem}"
        out_path = unique_output_path(dest_dir, out_name_base)
        cv2.imwrite(str(out_path), gray)

        metadata.append(
            {
                "split": target_split,
                "emotion": emotion_name,
                "image_path": str(out_path.relative_to(target_root)),
                "source_path": str(image_path.relative_to(source_root)),
                "landmarks": json.dumps(normalized_landmarks.tolist(), ensure_ascii=False),
            }
        )
        processed += 1

    logging.info("Processed %d samples from %s -> %s", processed, source_split, target_split)
    return metadata


def build_dataset(args):
    source_root = Path(args.source)
    target_root = Path(args.target)

    if not source_root.exists():
        raise FileNotFoundError(f"{source_root} does not exist")

    if target_root.exists() and args.overwrite:
        logging.info("Removing existing output at %s", target_root)
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    model = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device=args.device,
    )
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    metadata: list[dict[str, str]] = []
    for split in ("train", "valid", "test"):
        metadata.extend(
            process_split(
                model=model,
                detector=detector,
                source_root=source_root,
                source_split=split,
                target_root=target_root,
                target_size=args.resize,
                max_samples=args.max_per_split,
            )
        )

    metadata_path = target_root / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "emotion", "image_path", "source_path", "landmarks"],
        )
        writer.writeheader()
        writer.writerows(metadata)

    logging.info("Finished. Dataset ready at %s", target_root)
    return metadata_path


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare AffectNet into FER-style landmark-aware dataset")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/affectnet-yolo-format"),
        help="AffectNet YOLO root containing train/valid/test with images/ and labels/",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("data/affectnet-emotion-classifier-dataset"),
        help="Output root with FER-style train/test emotion folders and metadata.csv",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=64,
        help="Final square image size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for face_alignment (cpu/mps/cuda)",
    )
    parser.add_argument(
        "--max-per-split",
        type=int,
        default=None,
        help="Optional cap per source split for quick tests",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete target folder before processing",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
