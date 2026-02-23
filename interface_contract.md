# Interface Contract
Handwritten Capital Letter Recognition System
(Classical Computer Vision Only)

This document defines the required input and output format
for each module in the system.

All team members must strictly follow these specifications.

------------------------------------------------------------
1. DATA FORMAT (GLOBAL STANDARD)
------------------------------------------------------------

- Image type: numpy.ndarray
- Color format:
    - Segmentation input: BGR (from cv2.imread)
    - Preprocessing output: Binary (0 or 255)
- Binary rule:
    - Ink (letter) = 255 (white)
    - Background = 0 (black)
- Normalized image size: 32x32 pixels

------------------------------------------------------------
2. SEGMENTATION MODULE
------------------------------------------------------------

File: segmentation.py

Required function:

    segment_row_image(img_path: str) -> list[np.ndarray]

Input:
    - img_path: string path to row image
    - Image may contain multiple letters

Output:
    - List of cropped character images
    - Each element must be numpy.ndarray (BGR format)
    - Characters must be sorted left-to-right

Rules:
    - Use Otsu threshold or adaptive threshold
    - Use connectedComponentsWithStats
    - Filter small components by area
    - DO NOT resize inside segmentation

------------------------------------------------------------
3. PREPROCESSING MODULE
------------------------------------------------------------

File: preprocessing.py

Required function:

    preprocess_char(img: np.ndarray) -> np.ndarray

Input:
    - Single character image (BGR)

Output:
    - 32x32 binary image
    - Ink must be white (255)
    - Background must be black (0)
    - Character must be centered
    - dtype must be uint8

Rules:
    - Grayscale
    - Gaussian blur
    - Otsu threshold
    - Morphology cleaning
    - Tight bounding box crop
    - Pad to square
    - Resize to 32x32

------------------------------------------------------------
4. FEATURE EXTRACTION MODULE
------------------------------------------------------------

File: features.py

Required function:

    extract_features(img: np.ndarray) -> np.ndarray

Input:
    - 32x32 binary image (uint8)

Output:
    - 1D numpy array (feature vector)
    - dtype must be float32
    - Shape must be consistent for all samples

Examples:
    - Flatten: 1024 features
    - Zoning 8x8: 64 features
    - Projection: 64 features
    - Combined: 132 features

------------------------------------------------------------
5. CLASSIFIER MODULE
------------------------------------------------------------

File: classifier.py

Required functions:

    train_knn(X_train, y_train)
    knn_predict(X_train, y_train, X_test, k=3)

Input:
    - X_train: shape (N, D)
    - y_train: shape (N,)
    - X_test: shape (M, D)

Output:
    - Predictions: numpy array shape (M,)
    - Labels must be integer encoded (0–25)

Rules:
    - Must implement k-NN manually
    - Use Euclidean distance
    - No sklearn

------------------------------------------------------------
6. EVALUATION MODULE
------------------------------------------------------------

File: evaluation.py

Required functions:

    compute_accuracy(y_true, y_pred) -> float
    compute_confusion_matrix(y_true, y_pred) -> np.ndarray

Output:
    - Accuracy (0–1)
    - Confusion matrix shape (26, 26)

------------------------------------------------------------
7. FINAL PIPELINE
------------------------------------------------------------

File: main_pipeline.py

Required function:

    recognize_row_image(img_path: str) -> str

Process:
    1. Call segment_row_image
    2. For each character:
            preprocess_char
            extract_features
            knn_predict
    3. Convert numeric labels to letters
    4. Return final string

------------------------------------------------------------
8. DIRECTORY STRUCTURE
------------------------------------------------------------

project/
    segmentation.py
    preprocessing.py
    features.py
    classifier.py
    evaluation.py
    main_pipeline.py
    interface_contract.md

------------------------------------------------------------

All team members must follow this interface strictly.
If changes are needed, they must be discussed before modification.