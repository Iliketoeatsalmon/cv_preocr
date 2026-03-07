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


def _prepare_bgr_and_gray(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if img is None:
        raise ValueError("Input image is None")
    if img.ndim == 2:
        gray = img.copy()
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return bgr, gray
    if img.ndim == 3:
        return img.copy(), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported input image shape: {img.shape}")


def _find_component_boxes(bw: np.ndarray) -> list[tuple[int, int, int, int]]:
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

    boxes.sort(key=lambda b: b[0])
    return boxes


def segment_row_array_with_boxes(
    img: np.ndarray,
) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    """
    Segment a row image into left-to-right character crops (BGR images).
    Returns both the crops and their bounding boxes on the input image.
    """
    img_bgr, gray = _prepare_bgr_and_gray(img)
    h_img, w_img = img_bgr.shape[:2]

    # Single-character crops in this project are portrait-like; skip segmentation.
    if w_img <= int(1.4 * h_img):
        return [img_bgr], [(0, 0, w_img, h_img)]

    bw = _binarize_for_components(gray)
    boxes = _find_component_boxes(bw)

    if not boxes:
        return [img_bgr], [(0, 0, w_img, h_img)]

    crops: list[np.ndarray] = []
    padded_boxes: list[tuple[int, int, int, int]] = []
    for x, y, w, h in boxes:
        margin = 2
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(w_img, x + w + margin)
        y1 = min(h_img, y + h + margin)
        crops.append(img_bgr[y0:y1, x0:x1].copy())
        padded_boxes.append((x0, y0, x1 - x0, y1 - y0))

    return crops, padded_boxes


def segment_row_array(img: np.ndarray) -> list[np.ndarray]:
    crops, _ = segment_row_array_with_boxes(img)
    return crops


def segment_row_image(img_path: str) -> list[np.ndarray]:
    """
    Segment a row image into left-to-right character crops (BGR images).
    """
    path = Path(img_path)
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    return segment_row_array(img)
