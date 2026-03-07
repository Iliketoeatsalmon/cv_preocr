from __future__ import annotations

import numpy as np


def _encode_label(value: object) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        value = value.strip().upper()
        if len(value) == 1 and "A" <= value <= "Z":
            return ord(value) - ord("A")
    raise ValueError(f"Unsupported label value: {value!r}")


def _flatten_labels(labels) -> np.ndarray:
    return np.asarray(labels, dtype=object).reshape(-1)


def compute_accuracy(y_true, y_pred) -> float:
    y_true_arr = np.asarray([_encode_label(v) for v in _flatten_labels(y_true)])
    y_pred_arr = np.asarray([_encode_label(v) for v in _flatten_labels(y_pred)])
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true_arr.shape}, y_pred={y_pred_arr.shape}")
    return float(np.mean(y_true_arr == y_pred_arr))


def compute_confusion_matrix(y_true, y_pred, num_classes: int = 26) -> np.ndarray:
    y_true_arr = np.asarray([_encode_label(v) for v in _flatten_labels(y_true)])
    y_pred_arr = np.asarray([_encode_label(v) for v in _flatten_labels(y_pred)])
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true_arr.shape}, y_pred={y_pred_arr.shape}")

    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true_arr, y_pred_arr):
        if not (0 <= int(t) < num_classes) or not (0 <= int(p) < num_classes):
            raise ValueError(f"Label out of range for num_classes={num_classes}: true={t}, pred={p}")
        cm[int(t), int(p)] += 1
    return cm


def print_confusion_matrix(cm: np.ndarray) -> None:
    print("Confusion Matrix:")
    print(cm)


# Backward-compatible alias for old typo function name.
computer_accuracy = compute_accuracy


results_table: list[dict[str, object]] = []


def add_result(preprocess_name: str, feature_name: str, k: int, accuracy: float) -> None:
    results_table.append(
        {
            "preprocess": preprocess_name,
            "feature": feature_name,
            "k": int(k),
            "accuracy": float(accuracy),
        }
    )
