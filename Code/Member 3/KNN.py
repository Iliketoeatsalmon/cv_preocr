import cv2
import numpy as np

# ==========================================================
# FEATURE EXTRACTION (same functions as other file)
# ==========================================================

def flatten_feature(img):
    return img.flatten() / 255.0


def zoning_feature(img, grid_size=4):

    h, w = img.shape
    zone_h = h // grid_size
    zone_w = w // grid_size

    features = []

    for i in range(grid_size):
        for j in range(grid_size):

            zone = img[i*zone_h:(i+1)*zone_h,
                       j*zone_w:(j+1)*zone_w]

            density = np.sum(zone == 255) / (zone_h * zone_w)
            features.append(density)

    return np.array(features)


def projection_feature(img):

    horizontal = np.sum(img == 255, axis=1)
    vertical = np.sum(img == 255, axis=0)

    return np.concatenate((horizontal, vertical)) / 32


def extract_feature(img, method):

    if method == "flatten":
        return flatten_feature(img)

    elif method == "zoning":
        return zoning_feature(img)

    elif method == "projection":
        return projection_feature(img)


# ==========================================================
# KNN CLASSIFIER
# ==========================================================

def euclidean_distance(x1, x2):

    return np.sqrt(np.sum((x1 - x2) ** 2))


def knn_predict(X_train, y_train, x_test, k):

    distances = []

    for i in range(len(X_train)):

        dist = euclidean_distance(X_train[i], x_test)
        distances.append((dist, y_train[i]))

    distances.sort(key=lambda x: x[0])

    k_neighbors = distances[:k]

    labels = [label for (_, label) in k_neighbors]

    prediction = max(set(labels), key=labels.count)

    return prediction


# ==========================================================
# PREPROCESS IMAGE
# ==========================================================

def preprocess_image(path):

    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (32,32))

    _, binary = cv2.threshold(gray,0,255,
                              cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    if np.sum(binary == 255) < np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)

    return binary


# ==========================================================
# MAIN TEST
# ==========================================================

if __name__ == "__main__":

    test_path = r"Image Path"

    test_img = preprocess_image(test_path)

    # Feature method
    feature_method = "zoning"

    x_test = extract_feature(test_img, feature_method)

    # ------------------------------------------------------
    # Dummy training data (temporary)
    # ------------------------------------------------------

    X_train = [
        x_test + np.random.normal(0,0.05,len(x_test)),
        x_test + np.random.normal(0,0.05,len(x_test)),
        x_test + np.random.normal(0,0.05,len(x_test))
    ]

    y_train = ["A","A","A"]

    # ------------------------------------------------------
    # Compare different k values
    # ------------------------------------------------------

    k_values = [1,3,5]

    print("Feature method:", feature_method)

    for k in k_values:

        prediction = knn_predict(X_train,y_train,x_test,k)

        print(f"k = {k}  -> Prediction: {prediction}")