"""K-Means baseline for emotion dataset using the same data layout as the model evaluation script."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
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


RANDOM_SEED = 1
IMAGE_SIZE = 64
DEFAULT_K = len(CLASS_ORDER)
DEFAULT_ROOT = Path("data/emotion-classifier-dataset")

FILTER_TRANSFORMS = {
    "sobel": transforms.Lambda(apply_sobel_to_pil),
    "roberts": transforms.Lambda(apply_roberts_to_pil),
    "laplacian": transforms.Lambda(apply_laplacian_to_pil),
    "none": None,
}


def build_transform(filter_name: str) -> transforms.Compose:
    ops: list[transforms.Transform] = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    filter_transform = FILTER_TRANSFORMS.get(filter_name)
    if filter_transform is not None:
        ops.append(filter_transform)
    ops.extend([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    return transforms.Compose(ops)


def load_split(root: Path, split: str, filter_name: str) -> EmotionFolderWithPaths:
    dataset = EmotionFolderWithPaths(root=root / split, transform=build_transform(filter_name))
    filter_classes(dataset)
    return dataset


def collect_features(dataset: EmotionFolderWithPaths, batch_size: int, max_samples: int | None) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    seen = 0
    for images, targets, _ in tqdm(loader, desc="Extracting features", leave=False):
        batch_x = images.view(images.size(0), -1).cpu().numpy().astype(np.float32)
        batch_y = targets.cpu().numpy().astype(np.int64)

        if max_samples is not None and seen + batch_x.shape[0] > max_samples:
            keep = max_samples - seen
            if keep <= 0:
                break
            batch_x = batch_x[:keep]
            batch_y = batch_y[:keep]

        features.append(batch_x)
        labels.append(batch_y)
        seen += batch_x.shape[0]
        if max_samples is not None and seen >= max_samples:
            break

    if not features:
        return np.zeros((0, IMAGE_SIZE * IMAGE_SIZE), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def balanced_subsample(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    max_per_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    class_indices: dict[int, np.ndarray] = {}
    for cls in range(len(CLASS_ORDER)):
        idx = np.where(y == cls)[0]
        if idx.size > 0:
            class_indices[cls] = idx
    if not class_indices:
        return x, y

    target = min(indices.size for indices in class_indices.values())
    if max_per_class is not None:
        target = min(target, max_per_class)

    picked = []
    for cls in range(len(CLASS_ORDER)):
        indices = class_indices.get(cls)
        if indices is None or indices.size == 0:
            continue
        rng.shuffle(indices)
        picked.append(indices[:target])
    if not picked:
        return x, y

    selected = np.concatenate(picked)
    rng.shuffle(selected)
    return x[selected], y[selected]


def fit_pca_whitening(x: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0, keepdims=True)
    x_centered = x - mean

    # SVD-based PCA fit on training data only.
    _, singular_values, vt = np.linalg.svd(x_centered, full_matrices=False)
    d = min(n_components, vt.shape[0])
    components = vt[:d].T
    scales = singular_values[:d] / np.sqrt(max(x.shape[0] - 1, 1))
    return mean.astype(np.float32), components.astype(np.float32), scales.astype(np.float32)


def transform_pca_whitening(
    x: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    scales: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    projected = (x - mean) @ components
    return (projected / np.maximum(scales, eps)).astype(np.float32)


def kmeans_plus_plus_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    centroids = np.empty((k, x.shape[1]), dtype=np.float32)
    first = rng.integers(0, n)
    centroids[0] = x[first]

    closest_sq = np.sum((x - centroids[0]) ** 2, axis=1)
    for c in range(1, k):
        probs = closest_sq / np.maximum(closest_sq.sum(), 1e-12)
        idx = rng.choice(n, p=probs)
        centroids[c] = x[idx]
        dist_sq = np.sum((x - centroids[c]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, dist_sq)
    return centroids


def assign_clusters(x: np.ndarray, centroids: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    assignments = np.empty((x.shape[0],), dtype=np.int64)
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        chunk = x[start:end]
        dists = np.sum((chunk[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        assignments[start:end] = np.argmin(dists, axis=1)
    return assignments


def run_kmeans(
    x: np.ndarray,
    k: int,
    max_iter: int,
    tol: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if x.shape[0] < k:
        raise ValueError(f"Not enough samples ({x.shape[0]}) for k={k}.")

    rng = np.random.default_rng(seed)
    centroids = kmeans_plus_plus_init(x, k=k, rng=rng)
    assignments = np.zeros((x.shape[0],), dtype=np.int64)

    for it in range(max_iter):
        assignments = assign_clusters(x, centroids)
        new_centroids = np.zeros_like(centroids)
        counts = np.bincount(assignments, minlength=k)

        for c in range(k):
            if counts[c] == 0:
                new_centroids[c] = x[rng.integers(0, x.shape[0])]
            else:
                new_centroids[c] = x[assignments == c].mean(axis=0)

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        logging.info("K-Means iter %d/%d | shift=%.6f", it + 1, max_iter, shift)
        if shift <= tol:
            logging.info("Converged at iteration %d.", it + 1)
            break

    return centroids, assignments


def cluster_to_class_map(clusters: np.ndarray, labels: np.ndarray, k: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for c in range(k):
        mask = clusters == c
        if not np.any(mask):
            mapping[c] = 0
            continue
        counts = np.bincount(labels[mask], minlength=len(CLASS_ORDER))
        mapping[c] = int(np.argmax(counts))
    return mapping


def cluster_to_class_map_optimal(clusters: np.ndarray, labels: np.ndarray, k: int) -> dict[int, int]:
    num_classes = len(CLASS_ORDER)
    if k != num_classes:
        logging.warning(
            "Optimal 1:1 assignment requires k == num_classes (%d). Falling back to majority-vote mapping.",
            num_classes,
        )
        return cluster_to_class_map(clusters, labels, k)

    confusion = np.zeros((k, num_classes), dtype=np.int64)
    for c in range(k):
        mask = clusters == c
        if np.any(mask):
            confusion[c] = np.bincount(labels[mask], minlength=num_classes)

    best_score = -1
    best_perm: tuple[int, ...] | None = None
    for perm in itertools.permutations(range(num_classes)):
        score = int(sum(confusion[c, perm[c]] for c in range(k)))
        if score > best_score:
            best_score = score
            best_perm = perm

    if best_perm is None:
        return cluster_to_class_map(clusters, labels, k)
    return {c: int(best_perm[c]) for c in range(k)}


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def macro_precision(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    values = []
    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        denom = tp + fp
        values.append(float(tp / denom) if denom else 0.0)
    return float(np.mean(values))


def macro_recall(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    values = []
    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        denom = tp + fn
        values.append(float(tp / denom) if denom else 0.0)
    return float(np.mean(values))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    values = []
    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        values.append((2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0)
    return float(np.mean(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K-Means-only emotion clustering baseline.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT, help="Path to emotion-classifier-dataset")
    parser.add_argument(
        "--filter",
        choices=list(FILTER_TRANSFORMS.keys()),
        default="sobel",
        help="Edge filter applied before grayscale conversion.",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of clusters for K-Means.")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum K-Means iterations.")
    parser.add_argument("--tol", type=float, default=1e-3, help="Convergence threshold for centroid shift.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for feature extraction/assignment.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap for train samples (for faster experiments).",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional cap for test samples (for faster experiments).",
    )
    parser.add_argument(
        "--normalize-features",
        action="store_true",
        help="Apply L2 normalization to flattened image features before clustering.",
    )
    parser.add_argument(
        "--no-balance-train",
        action="store_true",
        help="Disable class-balanced subsampling of train features before K-Means.",
    )
    parser.add_argument(
        "--balance-max-per-class",
        type=int,
        default=None,
        help="Optional upper cap for samples per class when balancing train features.",
    )
    parser.add_argument(
        "--no-pca-whitening",
        action="store_true",
        help="Disable PCA + whitening feature projection before K-Means.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=128,
        help="Number of PCA components for whitening projection.",
    )
    parser.add_argument(
        "--no-optimal-assignment",
        action="store_true",
        help="Disable optimal 1:1 cluster-to-class assignment and use majority vote.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path to write metrics and cluster mapping as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.info("Loading dataset from %s (filter=%s)", args.data_root, args.filter)
    train_dataset = load_split(args.data_root, "train", args.filter)
    test_dataset = load_split(args.data_root, "test", args.filter)
    logging.info("Train samples: %d | Test samples: %d", len(train_dataset), len(test_dataset))

    x_train, y_train = collect_features(train_dataset, batch_size=args.batch_size, max_samples=args.max_train_samples)
    x_test, y_test = collect_features(test_dataset, batch_size=args.batch_size, max_samples=args.max_test_samples)

    if args.normalize_features:
        x_train = l2_normalize(x_train)
        x_test = l2_normalize(x_test)

    if not args.no_balance_train:
        before = x_train.shape[0]
        x_train, y_train = balanced_subsample(
            x_train,
            y_train,
            seed=args.seed,
            max_per_class=args.balance_max_per_class,
        )
        logging.info("Balanced train set: %d -> %d samples", before, x_train.shape[0])

    if not args.no_pca_whitening:
        max_rank = max(1, min(x_train.shape[0] - 1, x_train.shape[1]))
        pca_dim = min(max(1, args.pca_components), max_rank)
        mean, components, scales = fit_pca_whitening(x_train, pca_dim)
        x_train = transform_pca_whitening(x_train, mean, components, scales)
        x_test = transform_pca_whitening(x_test, mean, components, scales)
        logging.info("Applied PCA+whitening: %d -> %d dims", IMAGE_SIZE * IMAGE_SIZE, pca_dim)

    logging.info("Running K-Means (k=%d, max_iter=%d, tol=%g)", args.k, args.max_iter, args.tol)
    centroids, train_clusters = run_kmeans(
        x_train,
        k=args.k,
        max_iter=args.max_iter,
        tol=args.tol,
        seed=args.seed,
    )
    if args.no_optimal_assignment:
        mapping = cluster_to_class_map(train_clusters, y_train, args.k)
    else:
        mapping = cluster_to_class_map_optimal(train_clusters, y_train, args.k)

    test_clusters = assign_clusters(x_test, centroids, batch_size=args.batch_size)
    y_pred = np.array([mapping[c] for c in test_clusters], dtype=np.int64)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred) * 100.0,
        "precision": macro_precision(y_test, y_pred, len(CLASS_ORDER)),
        "recall": macro_recall(y_test, y_pred, len(CLASS_ORDER)),
        "f1": macro_f1(y_test, y_pred, len(CLASS_ORDER)),
    }

    logging.info(
        "Test metrics | acc=%.2f%% precision=%.4f recall=%.4f f1=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )

    train_cluster_counts = Counter(train_clusters.tolist())
    logging.info("Cluster sizes (train): %s", dict(sorted(train_cluster_counts.items())))
    readable_mapping = {int(k): CLASS_ORDER[v] for k, v in mapping.items()}
    logging.info("Cluster -> class mapping: %s", readable_mapping)

    if args.report_json is not None:
        payload = {
            "data_root": str(args.data_root),
            "filter": args.filter,
            "k": args.k,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "normalize_features": args.normalize_features,
            "balanced_train": not args.no_balance_train,
            "balance_max_per_class": args.balance_max_per_class,
            "pca_whitening": not args.no_pca_whitening,
            "pca_components": int(args.pca_components),
            "optimal_assignment": not args.no_optimal_assignment,
            "train_samples": int(x_train.shape[0]),
            "test_samples": int(x_test.shape[0]),
            "metrics": metrics,
            "cluster_sizes_train": {int(k): int(v) for k, v in train_cluster_counts.items()},
            "cluster_to_class": {int(k): CLASS_ORDER[v] for k, v in mapping.items()},
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logging.info("Saved report to %s", args.report_json)


if __name__ == "__main__":
    main()
