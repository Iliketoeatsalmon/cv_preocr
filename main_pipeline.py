from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from classifier import knn_predict, train_knn
from features import extract_features
from preprocessing import preprocess_char
from segmentation import segment_row_array_with_boxes


EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
_MODEL_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


@dataclass
class OCRResult:
    text: str
    predictions: list[str]
    char_crops: list[np.ndarray]
    normalized_chars: list[np.ndarray]
    boxes: list[tuple[int, int, int, int]]


def _load_training_dataset(dataset_root: Path) -> tuple[np.ndarray, np.ndarray]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root.resolve()}")

    X: list[np.ndarray] = []
    y: list[int] = []

    writer_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    for writer_dir in writer_dirs:
        for letter_dir in sorted([p for p in writer_dir.iterdir() if p.is_dir()]):
            letter = letter_dir.name.strip().upper()
            if len(letter) != 1 or not ("A" <= letter <= "Z"):
                continue
            label_id = ord(letter) - ord("A")
            for img_path in sorted(letter_dir.iterdir()):
                if img_path.suffix.lower() not in EXTS:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h_img, w_img = img.shape[:2]
                # Skip wide row-like images stored in folders (e.g., full-line scans).
                if w_img > int(1.4 * h_img):
                    continue
                norm = preprocess_char(img)
                feat = extract_features(norm)
                X.append(feat)
                y.append(label_id)

    if not X:
        raise RuntimeError(f"No valid training images found under {dataset_root.resolve()}")

    X_arr = np.vstack(X).astype(np.float32)
    y_arr = np.asarray(y, dtype=np.int32)
    return train_knn(X_arr, y_arr)


def _get_model(dataset_root: str) -> tuple[np.ndarray, np.ndarray]:
    key = str(Path(dataset_root).resolve())
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = _load_training_dataset(Path(dataset_root))
    return _MODEL_CACHE[key]


def _predict_char_images(
    char_images: list[np.ndarray],
    dataset_root: str,
    k: int,
) -> tuple[list[str], list[np.ndarray]]:
    X_train, y_train = _get_model(dataset_root)
    if not char_images:
        return [], []

    normalized_chars: list[np.ndarray] = []
    feats: list[np.ndarray] = []
    for char_img in char_images:
        norm = preprocess_char(char_img)
        normalized_chars.append(norm)
        feats.append(extract_features(norm))

    X_test = np.vstack(feats).astype(np.float32)
    pred_ids = knn_predict(X_train, y_train, X_test, k=k)
    predictions = [chr(int(pid) + ord("A")) for pid in pred_ids]
    return predictions, normalized_chars


def analyze_row_array(img: np.ndarray, dataset_root: str = "dataset", k: int = 3) -> OCRResult:
    char_crops, boxes = segment_row_array_with_boxes(img)
    predictions, normalized_chars = _predict_char_images(char_crops, dataset_root=dataset_root, k=k)
    return OCRResult(
        text="".join(predictions),
        predictions=predictions,
        char_crops=char_crops,
        normalized_chars=normalized_chars,
        boxes=boxes,
    )


def analyze_row_image(img_path: str, dataset_root: str = "dataset", k: int = 3) -> OCRResult:
    path = Path(img_path)
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return analyze_row_array(img, dataset_root=dataset_root, k=k)


def recognize_row_array(img: np.ndarray, dataset_root: str = "dataset", k: int = 3) -> str:
    return analyze_row_array(img, dataset_root=dataset_root, k=k).text


def recognize_row_image(img_path: str, dataset_root: str = "dataset", k: int = 3) -> str:
    return analyze_row_image(img_path, dataset_root=dataset_root, k=k).text


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recognize uppercase letters from a row image.")
    parser.add_argument("--image", required=True, help="Path to row image")
    parser.add_argument(
        "--dataset-root",
        default="dataset",
        help="Root training dataset directory (default: dataset)",
    )
    parser.add_argument("--k", type=int, default=3, help="k value for k-NN (default: 3)")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    text = recognize_row_image(args.image, dataset_root=args.dataset_root, k=args.k)
    print(text)
