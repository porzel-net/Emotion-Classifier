#!/usr/bin/env python3
"""Show only the first few slides from the test split with the Neconet predictions."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

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
DEFAULT_SLIDES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse a few slides of Neconet predictions.")
    parser.add_argument(
        "--weights",
        "--weights-dir",
        dest="weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Checkpoint file or directory.",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Root of the dataset.")
    parser.add_argument("--device", default=None, help="Force torch device (cuda, mps, cpu).")
    parser.add_argument("--max-slides", type=int, default=DEFAULT_SLIDES, help="How many pages to load.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--workers", type=int, default=2, help="Number of data loader workers.")
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
        self.fig.subplots_adjust(bottom=0.12)
        prev_ax = self.fig.add_axes([0.30, 0.02, 0.15, 0.06])
        next_ax = self.fig.add_axes([0.55, 0.02, 0.15, 0.06])
        self.prev_button = Button(prev_ax, "Prev")
        self.next_button = Button(next_ax, "Next")
        self.prev_button.on_clicked(self._on_prev_clicked)
        self.next_button.on_clicked(self._on_next_clicked)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._render()

    def _render(self):
        self.logger.info("Rendering slide %d/%d", self.page + 1, self.pages)
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

        self.status.set_text(
            f"Slide {self.page+1}/{self.pages} - keys: left/right, n/p, a/d, space/backspace"
        )
        self.fig.canvas.draw_idle()

    @staticmethod
    def _normalize_key(raw_key: str | None) -> str:
        if not raw_key:
            return ""
        key = raw_key.lower().strip()
        if "+" in key:
            key = key.split("+")[-1]
        return key

    def _next_page(self) -> None:
        if self.page < self.pages - 1:
            self.page += 1
            self._render()

    def _prev_page(self) -> None:
        if self.page > 0:
            self.page -= 1
            self._render()

    def _on_next_clicked(self, _event) -> None:
        self._next_page()

    def _on_prev_clicked(self, _event) -> None:
        self._prev_page()

    def _on_key(self, event):
        key = self._normalize_key(event.key)
        if key in ("right", "→", "pagedown", "pgdown", "n", "d", "space"):
            self._next_page()
        elif key in ("left", "←", "pageup", "pgup", "p", "a", "backspace"):
            self._prev_page()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%H:%M:%S")
    device = get_device(args.device)
    LOGGER.info("Using device %s", device)

    slides_to_load = max(1, args.max_slides)
    max_records = slides_to_load * PER_PAGE

    LOGGER.info("Building loader for split=test (limit=%d samples)", max_records)
    dataset, loader = build_loader(
        args.dataset_root,
        "test",
        batch_size=args.batch_size,
        workers=args.workers,
        max_samples=max_records,
    )
    LOGGER.info("Dataset has %d usable samples", len(dataset))

    model = build_model(args.weights, device, logger=LOGGER)

    LOGGER.info("Loading predictions for the first %d slides", slides_to_load)
    print("Loading predictions ...")
    records = predict_records(model, loader, device, limit=max_records)

    SimpleBrowser(records, logger=LOGGER)
    plt.tight_layout(rect=[0.0, 0.10, 1.0, 1.0])
    plt.show()


if __name__ == "__main__":
    main()
