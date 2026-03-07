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


def train_knn(X_train, y_train):
    """
    Prepare and validate training arrays for manual k-NN.
    Returns:
      X_train: float32 array of shape (N, D)
      y_train: int32 array of shape (N,)
    """
    X_arr = np.asarray(X_train, dtype=np.float32)
    y_arr_raw = np.asarray(y_train)

    if X_arr.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape {X_arr.shape}")

    if y_arr_raw.ndim != 1:
        y_arr_raw = y_arr_raw.reshape(-1)

    if X_arr.shape[0] != y_arr_raw.shape[0]:
        raise ValueError(
            f"X_train and y_train size mismatch: {X_arr.shape[0]} vs {y_arr_raw.shape[0]}"
        )

    y_arr = np.asarray([_encode_label(v) for v in y_arr_raw], dtype=np.int32)
    return X_arr, y_arr


def knn_predict(X_train, y_train, X_test, k=3):
    """
    Manual k-NN using Euclidean distance.
    Input:
      X_train: (N, D)
      y_train: (N,)
      X_test:  (M, D) or (D,)
    Output:
      predictions: (M,) int32
    """
    X_arr, y_arr = train_knn(X_train, y_train)
    X_test_arr = np.asarray(X_test, dtype=np.float32)
    if X_test_arr.ndim == 1:
        X_test_arr = X_test_arr.reshape(1, -1)
    if X_test_arr.ndim != 2:
        raise ValueError(f"X_test must be 1D or 2D, got shape {X_test_arr.shape}")
    if X_test_arr.shape[1] != X_arr.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: X_train={X_arr.shape[1]}, X_test={X_test_arr.shape[1]}"
        )

    if k <= 0:
        raise ValueError("k must be >= 1")
    k_use = min(int(k), X_arr.shape[0])

    diff = X_arr[None, :, :] - X_test_arr[:, None, :]
    dists = np.linalg.norm(diff, axis=2)  # (M, N)
    nn_idx = np.argpartition(dists, kth=k_use - 1, axis=1)[:, :k_use]

    preds: list[int] = []
    for row in nn_idx:
        labels = y_arr[row]
        uniq, counts = np.unique(labels, return_counts=True)
        max_count = counts.max()
        winners = uniq[counts == max_count]
        preds.append(int(winners.min()))

    return np.asarray(preds, dtype=np.int32)
