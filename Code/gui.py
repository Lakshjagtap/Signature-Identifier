import os
import tkinter as tk
from tkinter import filedialog, messagebox, Canvas, Frame, Scrollbar
from PIL import Image, ImageTk
import cv2
from feature_extractor import hog_descriptor
from model import load_model, train_model
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

IMG_SIZE = 128

class SignatureApp:
    def __init__(self, root, dataset_path=r"E:\MiniProject ML\Dataset"):
        self.root = root
        self.root.title("Signature Identification System")
        self.root.geometry("1200x700")
        self.root.minsize(1000, 600)  # Minimum size
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Load or train model
        if not os.path.exists("signature_identification_model.pkl"):
            self.clf, self.label_names, self.acc, self.cm, self.student_images = train_model(dataset_path)
        else:
            self.clf, self.label_names, self.acc, self.cm, self.student_images = load_model()

        # ---------------- Top Frame ----------------
        top_frame = tk.Frame(root, pady=5)
        top_frame.grid(row=0, column=0, sticky="ew")
        top_frame.columnconfigure(2, weight=1)  # Make image label expand

        self.upload_btn = tk.Button(top_frame, text="Upload Signature", font=("Arial", 12),
                                    command=self.upload_and_predict, bg="#4CAF50", fg="white")
        self.upload_btn.grid(row=0, column=0, padx=5)

        self.result_label = tk.Label(top_frame, text="", font=("Arial", 14, "bold"), fg="blue")
        self.result_label.grid(row=0, column=1, padx=10)

        self.img_label = tk.Label(top_frame)
        self.img_label.grid(row=0, column=2, padx=10, sticky="e")

        # ---------------- Bottom Frame ----------------
        bottom_frame = tk.Frame(root)
        bottom_frame.grid(row=1, column=0, sticky="nsew")
        bottom_frame.columnconfigure(0, weight=3)
        bottom_frame.columnconfigure(1, weight=1)
        bottom_frame.rowconfigure(0, weight=1)

        # ---------------- Matplotlib Performance Plots ----------------
        self.fig, self.axs = plt.subplots(1, 2, figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=bottom_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.plot_model_performance()

        # ---------------- Sample Signatures Panel ----------------
        sample_frame = tk.Frame(bottom_frame, bd=1, relief="sunken")
        sample_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        sample_frame.rowconfigure(1, weight=1)
        sample_frame.columnconfigure(0, weight=1)

        sample_label = tk.Label(sample_frame, text="Sample Signatures", font=("Arial", 12, "bold"))
        sample_label.grid(row=0, column=0, pady=5)

        # Scrollable sample signatures
        self.scroll_canvas = Canvas(sample_frame)
        self.scroll_canvas.grid(row=1, column=0, sticky="nsew")
        scrollbar = tk.Scrollbar(sample_frame, orient="vertical", command=self.scroll_canvas.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)
        self.sample_inner = tk.Frame(self.scroll_canvas)
        self.scroll_canvas.create_window((0,0), window=self.sample_inner, anchor='nw')
        self.sample_inner.bind("<Configure>", lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))
        self.display_sample_signatures()

        # ---------------- Misclassified Signatures Panel ----------------
        misclass_frame = tk.Frame(bottom_frame, bd=1, relief="sunken")
        misclass_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        misclass_frame.rowconfigure(1, weight=1)
        misclass_frame.columnconfigure(0, weight=1)

        misclass_label = tk.Label(misclass_frame, text="Misclassified Signatures", font=("Arial", 12, "bold"))
        misclass_label.grid(row=0, column=0, pady=5)

        self.scroll_misclass = Canvas(misclass_frame)
        self.scroll_misclass.grid(row=1, column=0, sticky="nsew")
        scrollbar_mis = Scrollbar(misclass_frame, orient="vertical", command=self.scroll_misclass.yview)
        scrollbar_mis.grid(row=1, column=1, sticky="ns")
        self.scroll_misclass.configure(yscrollcommand=scrollbar_mis.set)
        self.misclass_inner = Frame(self.scroll_misclass)
        self.scroll_misclass.create_window((0,0), window=self.misclass_inner, anchor='nw')
        self.misclass_inner.bind("<Configure>", lambda e: self.scroll_misclass.configure(scrollregion=self.scroll_misclass.bbox("all")))

        self.display_misclassified_signatures()


    # ------------------ Preprocess uploaded image ------------------
    def preprocess_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.GaussianBlur(img, (3, 3), 0)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        h = hog_descriptor.compute(img)
        return h.flatten().reshape(1, -1), img

    # ------------------ Upload & Predict ------------------
    def upload_and_predict(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            features, preprocessed_img = self.preprocess_image(file_path)
            if preprocessed_img is None:
                messagebox.showerror("Error", "Failed to read or preprocess image.")
                return

            # Confidence threshold
            threshold = 0.65  # 65%
            if hasattr(self.clf, "predict_proba"):
                proba = self.clf.predict_proba(features)[0]
                pred = np.argmax(proba)
                confidence = proba[pred] * 100
            else:
                # fallback if model does not support predict_proba
                pred = self.clf.predict(features)[0]
                confidence = None

            if confidence is not None and confidence < threshold * 100:
                result = "❌ Unknown / Forged Signature"
            else:
                result = f"✅ Predicted: {self.label_names[pred]}"
                if confidence is not None:
                    result += f" ({confidence:.2f}%)"

            self.result_label.config(text=result)

            # Display uploaded image
            img_pil = Image.fromarray(preprocessed_img).resize((150, 150))
            self.imgtk = ImageTk.PhotoImage(img_pil)
            self.img_label.configure(image=self.imgtk)


    # ------------------ Plot Accuracy & Confusion Matrix ------------------
    def plot_model_performance(self):
        self.axs[0].clear()
        self.axs[1].clear()

        # Accuracy bar
        self.axs[0].bar(["Accuracy"], [self.acc], color='green')
        self.axs[0].set_ylim(0, 1)
        self.axs[0].set_ylabel("Accuracy")
        self.axs[0].set_title("Model Accuracy")

        # Confusion matrix: adjust labels if needed
        num_classes = self.cm.shape[0]
        labels_present = self.label_names[:num_classes]

        disp = ConfusionMatrixDisplay(self.cm, display_labels=labels_present)
        disp.plot(ax=self.axs[1], cmap=plt.cm.Blues, colorbar=False)
        self.axs[1].set_title("Confusion Matrix")

        self.fig.tight_layout()
        self.canvas.draw()

    # ------------------ Display Sample Signatures ------------------
    def display_sample_signatures(self):
        for student, imgs in self.student_images.items():
            label = tk.Label(self.sample_inner, text=student, font=("Arial", 10, "bold"))
            label.pack(pady=2)
            grid_frame = tk.Frame(self.sample_inner)
            grid_frame.pack(pady=1)
            for i, img in enumerate(imgs[:6]):  # Show up to 6 samples per student
                img_pil = Image.fromarray(img).resize((60, 60))
                imgtk = ImageTk.PhotoImage(img_pil)
                img_label = tk.Label(grid_frame, image=imgtk)
                img_label.image = imgtk
                img_label.grid(row=0, column=i, padx=2)
                
    def display_misclassified_signatures(self):
        """
        Shows signatures that the model predicted incorrectly
        """
        from feature_extractor import hog_descriptor

        # Clear previous widgets
        self.misclass_inner.destroy()
        self.misclass_inner = Frame(self.scroll_misclass)
        self.scroll_misclass.create_window((0,0), window=self.misclass_inner, anchor='nw')

        for student_idx, student_name in enumerate(self.label_names):
            imgs = self.student_images[student_name]
            for img in imgs:
                h = hog_descriptor.compute(img).flatten().reshape(1, -1)
                pred = self.clf.predict(h)[0]

                # Safety check: ignore invalid predictions
                if pred >= len(self.label_names) or pred < 0:
                    continue

                if pred != student_idx:
                    # Display misclassified image
                    img_pil = Image.fromarray(img).resize((60, 60))
                    imgtk = ImageTk.PhotoImage(img_pil)
                    img_label = tk.Label(self.misclass_inner, image=imgtk)
                    img_label.image = imgtk
                    img_label.pack(pady=2)

                    info_label = tk.Label(
                        self.misclass_inner,
                        text=f"T:{student_name}\nP:{self.label_names[pred]}",
                        font=("Arial", 8)
                    )
                    info_label.pack(pady=2)
