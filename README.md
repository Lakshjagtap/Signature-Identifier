✍️ Signature Identifier

A "Signature Identification System" powered by Python, OpenCV, and scikit-learn.  
This project uses **image preprocessing, HOG feature extraction, and machine learning classifiers** to identify the signer of a given signature.  

An "interactive GUI (Tkinter)" is included for easy experimentation:
- Upload a signature & get predictions with confidence scores
- View accuracy charts & confusion matrix
- Browse sample signatures per student
- Inspect misclassified signatures for debugging

## ✨ Features
- 🖼 "Image Preprocessing": noise removal, Gaussian blur, Otsu thresholding  
- 🔄 "Data Augmentation": rotations, flips for robust training  
- 📐 "Feature Extraction": HOG (Histogram of Oriented Gradients)  
- 🤖 "Model Training": SVM, Random Forest, KNN, and MLP (best selected via cross-validation)  
- 🖥 "Graphical User Interface":
  - Upload signatures & preview them
  - Display prediction with confidence score
  - Visualize accuracy and confusion matrix
  - Explore genuine & misclassified samples

📂 Project Structure
Signature-Identifier/
── data_loader.py # Dataset loading & preprocessing
── feature_extractor.py # HOG feature extractor
── model.py # Model training, evaluation & saving
── gui.py # Tkinter-based GUI
── main.py # Entry point (checks model + launches GUI)
── requirements.txt # Dependencies
── README.md # Documentation
── signature_identification_model.pkl # Trained model (auto-generated)

⚙️ Installation

1. "Clone the repository" 
   git clone https://github.com/Lakshjagtap/Signature-Identifier.git
   
       cd Signature-Identifier
   
3.     pip install -r requirements.txt
4. Prepare the dataset
  Organize signature images as follows:

  Dataset/
  ├── Student1/
  │   ├── sig1.jpg
  │   ├── sig2.jpg
  ├── Student2/
  │   ├── sig1.jpg
  │   ├── sig2.jpg
  ...
  
▶️ Usage
🔹 Train the Model (optional)

If no model exists, it will be trained automatically.
To manually train:

    python model.py

🔹 Launch the GUI

    python main.py

🔹 In the GUI, you can:
Upload and preview signatures
Get predictions with confidence scores
View performance charts (accuracy, confusion matrix)
Browse sample signatures per student
Analyze misclassified cases
