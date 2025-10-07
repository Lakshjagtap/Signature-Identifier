import cv2
import numpy as np

IMG_SIZE = 128

# HOG descriptor configuration
hog_descriptor = cv2.HOGDescriptor(
    _winSize=(IMG_SIZE, IMG_SIZE),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)

def extract_hog_features(images):
    """
    Extract HOG features from a list or numpy array of grayscale images.

    Parameters:
        images (list or np.ndarray): List or array of 2D grayscale images (shape: IMG_SIZE x IMG_SIZE)

    Returns:
        np.ndarray: Array of HOG feature vectors
    """
    features = []
    for idx, img in enumerate(images):
        # Ensure image is grayscale and correct size
        if img.shape != (IMG_SIZE, IMG_SIZE):
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        if len(img.shape) != 2:
            raise ValueError(f"Image at index {idx} is not grayscale.")

        h = hog_descriptor.compute(img)
        features.append(h.flatten())

    return np.array(features)
