# Evaluation functions for the KNN classifier
import numpy as np

def computer_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def compute_confusion_matrix(y_true, y_pred, num_classes = 26):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def print_confusion_matrix(cm):
    print("Confusion Matrix:")
    print(cm)


results_table = []

def add_result(preprocess_name, feature_name, k, accuracy):
    results_table.append({
        "preprocess": preprocess_name,
        "feature": feature_name,
        "k": k,
        "accuracy": accuracy
    })