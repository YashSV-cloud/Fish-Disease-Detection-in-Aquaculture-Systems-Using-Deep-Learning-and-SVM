PROJECT TITLE
-------------
Fish Disease Detection in Aquaculture Systems Using Deep Learning and SVM


PROJECT DESCRIPTION
-------------------
This project is a machine learning–based web application designed to detect
fish diseases from images. It uses deep learning for feature extraction and
a Support Vector Machine (SVM) classifier for disease prediction.

The system helps in early detection of fish diseases in aquaculture systems,
reducing manual inspection and improving productivity.


FEATURES
--------
- Image-based fish disease detection
- Deep feature extraction using pretrained VGG16
- Disease classification using SVM
- Real-time prediction using Flask web application
- Simple and user-friendly interface


TECHNOLOGIES USED
-----------------
- Programming Language: Python
- Deep Learning: TensorFlow, Keras (VGG16)
- Machine Learning: Scikit-learn (SVM)
- Web Framework: Flask
- Libraries: NumPy, Pillow, Joblib


PROJECT WORKFLOW
----------------
1. User uploads a fish image through the web interface
2. Image is resized and normalized
3. VGG16 extracts deep features from the image
4. Extracted features are passed to the trained SVM model
5. The predicted fish disease is displayed to the user


PROJECT STRUCTURE
-----------------
Fish-Disease-Detection/
|
|-- app.py                  -> Flask web application
|-- model.py                -> Model training script
|-- requirements.txt        -> Python dependencies
|-- index.html              -> Frontend user interface
|-- datset                  -> Access can be provided upon request for academic or learning purposes
|-- lable_encoder           -> trained LabelEncoder object
|-- svm_fish_disease        -> Trained SVM disease classification model

DATASET INFORMATION
-------------------
The dataset contains images of freshwater fish affected by different disease
conditions. Images are organized into class-wise folders, where each folder
represents a specific fish disease.

Due to size limitations, the dataset is not included in this GitHub repository.
Download the dataset from Google Drive:
https://drive.google.com/drive/folders/12-YUP3FxFJS2OwK3nESBSWRd3D7YSgtO?usp=drive_link


DATASET ACCESS
--------------
The dataset is stored in a private Google Drive folder.
Access can be provided upon request for academic or learning purposes.


MODEL FILES
-----------
- svm_fish_disease.pkl     -> Trained SVM classification model
- label_encoder.pkl        -> Encodes disease class labels

Note:
## Model Files

Due to GitHub file size limitations, trained model files are not included in this repository.

Download the model files from Google Drive:
https://drive.google.com/drive/folders/12-YUP3FxFJS2OwK3nESBSWRd3D7YSgtO?usp=drive_link

After downloading, place the files in the project root directory before running the application.



INSTALLATION AND SETUP
----------------------
Step 1: Create virtual environment
python -m venv venv

Step 2: Activate virtual environment (Windows)
.\venv\Scripts\Activate.ps1

Step 3: Install dependencies
pip install --upgrade pip
pip install -r requirements.txt


RUNNING THE APPLICATION
-----------------------
python app.py

Open your browser and visit:
http://localhost:5000/

MODEL DETAILS
-------------
- Feature Extractor: VGG16 (pretrained on ImageNet)
- Classifier: Support Vector Machine (SVM)
- Input Image Size: 224 x 224
- Output: Fish disease class


FUTURE ENHANCEMENTS
-------------------
- Train an end-to-end CNN model
- Improve accuracy using data augmentation
- Add Docker support
- Deploy the application on cloud platforms


AUTHOR
------
Yashwanth SV
BCA Graduate
DevOps Student


DISCLAIMER
----------
This project is developed strictly for academic and learning purposes.
