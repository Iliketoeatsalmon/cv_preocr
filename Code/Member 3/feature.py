import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# LOAD AND PREPROCESS IMAGE
# ==========================================

img = cv2.imread(r"Image Path")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (32, 32))

_, binary = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# ensure white character
if np.sum(binary == 255) < np.sum(binary == 0):
    binary = cv2.bitwise_not(binary)

# ==========================================
# FEATURE EXTRACTION
# ==========================================

def flatten_feature(img):
    return img.flatten()


def zoning_feature(img, grid_size=4):

    h, w = img.shape
    zone_h = h // grid_size
    zone_w = w // grid_size

    features = []

    for i in range(grid_size):
        for j in range(grid_size):

            zone = img[i*zone_h:(i+1)*zone_h,
                       j*zone_w:(j+1)*zone_w]

            density = np.sum(zone == 255)
            features.append(density)

    return np.array(features)


def projection_feature(img):

    horizontal = np.sum(img == 255, axis=1)
    vertical = np.sum(img == 255, axis=0)

    return horizontal, vertical


# ==========================================
# EXTRACT FEATURES
# ==========================================

flatten = flatten_feature(binary)
zoning = zoning_feature(binary)
horizontal, vertical = projection_feature(binary)

# reshape flatten for visualization
flatten_img = flatten.reshape(32, 32)

# reshape zoning to grid
zoning_grid = zoning.reshape(4, 4)

# ==========================================
# PLOT RESULTS
# ==========================================

plt.figure(figsize=(12,8))

# original image
plt.subplot(2,2,1)
plt.imshow(binary, cmap="gray")
plt.title("Binary Image")
plt.axis("off")

# flatten
plt.subplot(2,2,2)
plt.imshow(flatten_img, cmap="gray")
plt.title("Flatten Feature Representation")
plt.axis("off")

# zoning
plt.subplot(2,2,3)
plt.imshow(zoning_grid, cmap="gray")
plt.title("Zoning Feature (4x4)")
plt.colorbar()

# projection histogram
plt.subplot(2,2,4)
plt.plot(horizontal, label="Horizontal")
plt.plot(vertical, label="Vertical")
plt.title("Projection Histogram")
plt.legend()

plt.tight_layout()
plt.show()