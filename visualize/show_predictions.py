#!/usr/bin/env python3
"""Interactive viewer for the entire Neconet test split."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from helpers.neconet_helpers import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_WEIGHTS,
    EMOTION_LABELS,
    build_loader,
    build_model,
    get_device,
    predict_records,
)

LOGGER = logging.getLogger(__name__)
PER_PAGE = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse Neconet test predictions.")
    parser.add_argument(
        "--weights",
        "--weights-dir",
        dest="weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to weights directory or file.",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Dataset root directory.")
    parser.add_argument("--device", default=None, help="Force a torch device (cuda, mps, cpu).")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--workers", type=int, default=2, help="Data loader workers.")
    return parser.parse_args()


class SimpleBrowser:
    def __init__(self, records: list[tuple[str, int, int, float]], logger: logging.Logger | None = None):
        self.logger = logger or LOGGER
        self.records = records
        self.per_page = PER_PAGE
        self.pages = max(1, (len(records) + self.per_page - 1) // self.per_page)
        self.page = 0
        self.fig, axes = plt.subplots(3, 3, figsize=(6, 6))
        self.axes = axes.flatten()
        self.status = self.fig.text(0.5, 0.95, "", ha="center", va="center")
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._render()

    def _render(self):
        self.logger.info("Rendering page %d/%d", self.page + 1, self.pages)
        self.status.set_text("Loading …")
        self.fig.canvas.draw_idle()

        start = self.page * self.per_page
        chunk = self.records[start : start + self.per_page]
        for ax, slot in zip(self.axes, chunk):
            ax.clear()
            img_path, truth, pred, conf = slot
            image = plt.imread(img_path)
            if image.ndim == 3:
                image = image[..., 0]
            ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{EMOTION_LABELS[truth]} → {EMOTION_LABELS[pred]} ({conf*100:4.1f}%)", fontsize=8)
            ax.axis("off")
        for ax in self.axes[len(chunk) :]:
            ax.clear()
            ax.axis("off")

        self.status.set_text(f"Page {self.page+1}/{self.pages} – ←/→ or n/p to browse")
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        key = (event.key or "").lower()
        if key in ("right", "→", "pagedown", "pgdown", "n"):
            if self.page < self.pages - 1:
                self.page += 1
                self._render()
        elif key in ("left", "←", "pageup", "pgup", "p"):
            if self.page > 0:
                self.page -= 1
                self._render()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%H:%M:%S")
    device = get_device(args.device)
    LOGGER.info("Using device %s", device)

    dataset, loader = build_loader(
        args.dataset_root,
        "test",
        batch_size=args.batch_size,
        workers=args.workers,
    )
    LOGGER.info("Test split contains %d filtered samples", len(dataset))

    model = build_model(args.weights, device, logger=LOGGER)

    LOGGER.info("Loading predictions for %d samples", len(dataset))
    records = predict_records(model, loader, device)

    SimpleBrowser(records, logger=LOGGER)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
