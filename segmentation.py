from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _binarize_for_components(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    kernel = np.ones((2, 2), dtype=np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    return bw


def segment_row_image(img_path: str) -> list[np.ndarray]:
    """
    Segment a row image into left-to-right character crops (BGR images).
    """
    path = Path(img_path)
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    h_img, w_img = img.shape[:2]
    # Single-character crops in this project are portrait-like; skip segmentation.
    if w_img <= int(1.4 * h_img):
        return [img]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = _binarize_for_components(gray)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    min_area = max(20, int(0.001 * bw.shape[0] * bw.shape[1]))

    boxes: list[tuple[int, int, int, int]] = []
    for comp_idx in range(1, num_labels):
        x, y, w, h, area = stats[comp_idx]
        if area < min_area:
            continue
        if w < 3 or h < 5:
            continue
        boxes.append((x, y, w, h))

    if not boxes:
        return [img]

    boxes.sort(key=lambda b: b[0])
    crops: list[np.ndarray] = []
    for x, y, w, h in boxes:
        margin = 2
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(w_img, x + w + margin)
        y1 = min(h_img, y + h + margin)
        crops.append(img[y0:y1, x0:x1].copy())

    return crops
