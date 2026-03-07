from __future__ import annotations

import cv2
import numpy as np


def preprocess_char(img: np.ndarray) -> np.ndarray:
    """
    Convert a character crop (BGR or grayscale) into a centered 32x32 binary image.
    Output convention:
      - ink (foreground): 255
      - background: 0
      - dtype: uint8
    """
    if img is None:
        raise ValueError("Input image is None")

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        gray = img.copy()
    else:
        raise ValueError(f"Unsupported input image shape: {img.shape}")

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    open_kernel = np.ones((2, 2), dtype=np.uint8)
    close_kernel = np.ones((2, 2), dtype=np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, open_kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    coords = cv2.findNonZero(bw)
    if coords is None:
        return np.zeros((32, 32), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    cropped = bw[y : y + h, x : x + w]

    ch, cw = cropped.shape
    side = max(ch, cw)
    pad = max(2, side // 8)
    canvas_size = side + 2 * pad

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    y0 = (canvas_size - ch) // 2
    x0 = (canvas_size - cw) // 2
    canvas[y0 : y0 + ch, x0 : x0 + cw] = cropped

    resized = cv2.resize(canvas, (32, 32), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY)
    return binary.astype(np.uint8)
