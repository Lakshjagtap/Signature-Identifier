import os
import cv2
import numpy as np
from sklearn.utils import shuffle

IMG_SIZE = 128

def load_dataset(root_folder):
    images, labels = [], []
    label_names = []
    student_images = {}

    if not os.path.exists(root_folder):
        raise FileNotFoundError(f"Dataset folder not found: {root_folder}")

    for idx, student in enumerate(os.listdir(root_folder)):
        student_folder = os.path.join(root_folder, student)
        if not os.path.isdir(student_folder):
            continue
        label_names.append(student)
        student_images[student] = []

        for filename in os.listdir(student_folder):
            if not filename.lower().endswith(('.png','.jpg','.jpeg')):
                continue

            img_path = os.path.join(student_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.GaussianBlur(img, (3,3), 0)
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            student_images[student].append(img.copy())

            # Original
            images.append(img)
            labels.append(idx)

            # Rotation augmentation
            for angle in [-10,10]:
                M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), angle, 1)
                rotated = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderValue=255)
                images.append(rotated)
                labels.append(idx)

            # Horizontal flip
            flipped = cv2.flip(img, 1)
            images.append(flipped)
            labels.append(idx)

    images, labels = shuffle(images, labels, random_state=42)
    return np.array(images), np.array(labels), label_names, student_images
