from __future__ import annotations

import cv2
import numpy as np


def _to_binary_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim != 2:
        raise ValueError(f"Expected 2D binary image, got shape {img.shape}")
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return (img > 0).astype(np.uint8) * 255


def _zoning_features(img: np.ndarray, grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    h, w = img.shape
    gy, gx = grid
    cell_h = h // gy
    cell_w = w // gx
    feats: list[float] = []
    for iy in range(gy):
        for ix in range(gx):
            cell = img[iy * cell_h : (iy + 1) * cell_h, ix * cell_w : (ix + 1) * cell_w]
            density = float(np.count_nonzero(cell)) / float(cell.size)
            feats.append(density)
    return np.asarray(feats, dtype=np.float32)


def _projection_features(img: np.ndarray) -> np.ndarray:
    fg = (img > 0).astype(np.float32)
    horiz = fg.sum(axis=1) / float(img.shape[1])
    vert = fg.sum(axis=0) / float(img.shape[0])
    return np.concatenate([horiz, vert]).astype(np.float32)


def _global_shape_features(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    density = np.count_nonzero(img) / float(img.size)
    coords = cv2.findNonZero(img)
    if coords is None:
        aspect = 1.0
        norm_h = 0.0
        norm_w = 0.0
    else:
        x, y, bw, bh = cv2.boundingRect(coords)
        _ = x, y
        aspect = (bw / bh) if bh > 0 else 1.0
        norm_h = bh / float(h)
        norm_w = bw / float(w)
    return np.asarray([density, aspect, norm_h, norm_w], dtype=np.float32)


def extract_features(img: np.ndarray) -> np.ndarray:
    """
    Extract a fixed 132-dim feature vector:
      - zoning 8x8: 64
      - projection: 64
      - global shape: 4
    """
    binary = _to_binary_uint8(img)
    if binary.shape != (32, 32):
        raise ValueError(f"Expected 32x32 input, got {binary.shape}")
    z = _zoning_features(binary, grid=(8, 8))
    p = _projection_features(binary)
    g = _global_shape_features(binary)
    return np.concatenate([z, p, g]).astype(np.float32)
