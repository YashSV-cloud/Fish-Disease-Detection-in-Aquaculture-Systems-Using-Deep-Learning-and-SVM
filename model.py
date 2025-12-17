import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Paths
dataset_path = "/content/datasets/fish_disease/Freshwater Fish Disease Aquaculture in south asia/"
train_img_dir = os.path.join(dataset_path, "Train")  # Training images directory

X, y = [], []

# Load Pretrained Model (VGG16 Only)
vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Feature Extraction Function (Using Only VGG16)
def extract_features(model, img_path):
    try:
        img = load_img(img_path, target_size=(224, 224))  # Load & Resize Image
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)
        features = model.predict(img_array)
        return features.flatten()
    except Exception as e:
        print(f"❌ Error processing image {img_path}: {e}")
        return None

# Read Images from Train Folder
for class_name in os.listdir(train_img_dir):  # Each folder is a class
    class_path = os.path.join(train_img_dir, class_name)

    if os.path.isdir(class_path):  # Ensure it's a folder
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            if os.path.exists(img_path):  # Ensure file exists
                vgg_features = extract_features(vgg16, img_path)

                if vgg_features is not None:
                    X.append(vgg_features)  # Use only VGG16 features
                    y.append(class_name)  # Folder name as label

# Convert to NumPy Arrays
X = np.array(X)
y = np.array(y)

# Ensure X and y are not empty
if len(X) == 0 or len(y) == 0:
    raise ValueError("❌ ERROR: No images were processed. Check dataset paths.")

# Encode Labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train SVM Classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Save Model & Label Encoder
joblib.dump(svm_model, "svm_fish_disease.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Model training complete and saved!")
