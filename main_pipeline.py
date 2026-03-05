def recognize_row_image(img_path):

    chars = segment_row_image(img_path)

    results = []

    for ch in chars:
        norm = preprocess_char(ch)
        feat = extract_features(norm)
        pred = knn_predict(X_train, y_train, feat[np.newaxis, :], k=3)
        results.append(pred)

    return results