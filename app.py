from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import joblib
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask App
app = Flask(__name__)

# Load Pretrained Model for Feature Extraction (VGG16 Only)
vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Load Trained SVM Model & Label Encoder
svm_model = joblib.load("svm_fish_disease.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Function to Extract Features from Image
def extract_features(img_path):
    img = load_img(img_path, target_size=(224, 224))  # Load & Resize Image
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    features = vgg16.predict(img_array)
    return features.flatten()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]  # Get Uploaded File
        if file:
            filepath = os.path.join("static", "uploaded.jpg")
            file.save(filepath)  # Save Image

            # Extract Features
            features = extract_features(filepath)

            # Predict
            prediction = svm_model.predict([features])
            predicted_class = label_encoder.inverse_transform(prediction)[0]

            return render_template("index.html", prediction=predicted_class, img_path=filepath)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
