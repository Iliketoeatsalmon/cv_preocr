from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from main_pipeline import OCRResult, analyze_row_array


DEFAULT_DATASET_ROOT = str((Path(__file__).resolve().parent / "dataset").resolve())


def _decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode the uploaded image.")
    return img


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _draw_predictions(
    img_bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    predictions: list[str],
) -> np.ndarray:
    canvas = img_bgr.copy()
    for idx, (box, pred) in enumerate(zip(boxes, predictions), start=1):
        x, y, w, h = box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 170, 255), 2)
        label = f"{idx}:{pred}"
        text_y = max(22, y - 8)
        cv2.putText(
            canvas,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 80, 0),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _show_crops(title: str, images: list[np.ndarray], captions: list[str], grayscale: bool = False) -> None:
    st.subheader(title)
    if not images:
        st.info("No character images to display.")
        return

    cols_per_row = min(4, len(images))
    for start in range(0, len(images), cols_per_row):
        cols = st.columns(cols_per_row)
        batch_images = images[start : start + cols_per_row]
        batch_captions = captions[start : start + cols_per_row]
        for col, img, caption in zip(cols, batch_images, batch_captions):
            with col:
                if grayscale:
                    st.image(img, clamp=True, use_container_width=True)
                else:
                    st.image(_bgr_to_rgb(img), use_container_width=True)
                st.caption(caption)


def _run_ocr(img_bgr: np.ndarray, dataset_root: str, k: int) -> OCRResult:
    with st.spinner("Training and reading the image..."):
        return analyze_row_array(img_bgr, dataset_root=dataset_root, k=k)


st.set_page_config(page_title="OCR Upload UI", layout="wide")

st.title("Handwritten OCR Upload UI")
st.caption("Upload a handwritten uppercase image, then the app will segment each character and show the predicted result.")

with st.sidebar:
    st.header("Settings")
    dataset_root = st.text_input("Dataset path", value=DEFAULT_DATASET_ROOT)
    k_value = st.slider("k for k-NN", min_value=1, max_value=9, value=3, step=1)
    st.caption("The first run will train the k-NN model from the dataset folder and cache it in memory.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "bmp"],
    help="Use a single character image or a row image containing multiple uppercase letters.",
)

if uploaded_file is None:
    st.info("Upload an image to start OCR.")
    st.stop()

try:
    uploaded_image = _decode_uploaded_image(uploaded_file.getvalue())
except Exception as exc:
    st.error(f"Unable to open this file: {exc}")
    st.stop()

try:
    result = _run_ocr(uploaded_image, dataset_root=dataset_root, k=k_value)
except Exception as exc:
    st.error(f"OCR failed: {exc}")
    st.stop()

annotated_image = _draw_predictions(uploaded_image, result.boxes, result.predictions)

preview_col, detected_col = st.columns(2)
with preview_col:
    st.subheader("Original Image")
    st.image(_bgr_to_rgb(uploaded_image), use_container_width=True)
with detected_col:
    st.subheader("Detected Characters")
    st.image(_bgr_to_rgb(annotated_image), use_container_width=True)

st.subheader("Prediction")
st.code(result.text or "(no text detected)", language="text")
st.write(f"Detected {len(result.predictions)} character(s).")

crop_captions = [f"Character {idx}: {pred}" for idx, pred in enumerate(result.predictions, start=1)]
normalized_captions = [
    f"Normalized {idx}: {pred}" for idx, pred in enumerate(result.predictions, start=1)
]

_show_crops("Segmented Crops", result.char_crops, crop_captions)
_show_crops("Normalized 32x32 Inputs", result.normalized_chars, normalized_captions, grayscale=True)
