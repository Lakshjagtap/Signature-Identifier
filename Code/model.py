import os
import cv2
import numpy as np
import joblib
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from data_loader import load_dataset
from feature_extractor import hog_descriptor, IMG_SIZE

MODEL_FILE = "signature_identification_model.pkl"

# ----------------------- Preprocessing -----------------------
def preprocess_image(img):
    """
    Denoise and binarize the image
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    img_gray = cv2.fastNlMeansDenoising(img_gray, h=10)
    img_bin = cv2.adaptiveThreshold(img_gray, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)
    img_bin = cv2.resize(img_bin, (IMG_SIZE, IMG_SIZE))
    return img_bin

def extract_features_with_preprocessing(images):
    """
    Apply preprocessing and extract HOG features
    """
    features = []
    for img in images:
        img_proc = preprocess_image(img)
        h = hog_descriptor.compute(img_proc)
        features.append(h.flatten())
    return np.array(features)

# ----------------------- Training -----------------------
def train_model(dataset_path):
    print("🔹 Loading dataset...")
    X, y, label_names, student_images = load_dataset(dataset_path)

    # Convert images to numpy arrays for preprocessing
    X = [np.array(img) for img in X]

    print("🔹 Extracting features with preprocessing...")
    X_features = extract_features_with_preprocessing(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )

    # ----------------------- Model Selection -----------------------
    models = {
        "SVM": svm.SVC(kernel='rbf', probability=True),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "MLP": MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500, random_state=42)
    }

    best_model = None
    best_acc = 0

    print("🔹 Training multiple models...")
    for name, model in models.items():
        # Cross-validation
        cv = StratifiedKFold(n_splits=5)
        scores = cross_val_score(model, X_features, y, cv=cv)
        mean_acc = np.mean(scores)
        print(f"{name} CV Accuracy: {mean_acc*100:.2f}%")

        # Train on full training set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        print(f"{name} Test Accuracy: {test_acc*100:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = model
            best_cm = confusion_matrix(y_test, y_pred)

    print(f"✅ Best model: {best_model.__class__.__name__} with Accuracy: {best_acc*100:.2f}%")

    # Save model
    joblib.dump((best_model, label_names, best_acc, best_cm, student_images), MODEL_FILE)
    print(f"💾 Model saved to {MODEL_FILE}")

    return best_model, label_names, best_acc, best_cm, student_images

# ----------------------- Load Model -----------------------
def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("Model not found. Please train first using train_model().")
    return joblib.load(MODEL_FILE)
