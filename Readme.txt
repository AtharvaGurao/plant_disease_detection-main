 Plant Disease Detection System

Overview

The Plant Disease Detection System is a machine learning-powered application designed to detect and diagnose plant diseases from leaf images. It uses a Convolutional Neural Network (CNN) model trained on a dataset of plant leaves with various diseases, providing accurate disease detection and remedies for farmers. The system is designed to offer early detection and actionable solutions, which helps in optimizing agricultural productivity by reducing crop losses and improving crop yields.

The system allows users to upload an image of a plant leaf through a web-based interface. After uploading the image, the system processes it and classifies it into one of several disease categories or identifies it as healthy. Additionally, the system provides remedies or treatment advice for the detected disease.

---

Features
- Upload images of plant leaves through a user-friendly web interface.
- Leverages a pre-trained Convolutional Neural Network (CNN) model to detect plant diseases.
- Provides remedies for the detected diseases to assist farmers with treatment.
- Displays the uploaded image with a bounding box indicating the affected area.
- Fast, accurate, and easy-to-use solution for plant disease detection.

---

Technology Stack

Backend
- Flask: A lightweight Python web framework to handle the web server, routing, and image uploading.
- TensorFlow/Keras: A deep learning library used to load and run the pre-trained CNN model for plant disease classification.

Frontend
- HTML, CSS, and Bootstrap for building a simple user interface.

Libraries/Tools Used
- PIL (Python Imaging Library): For image manipulation and drawing bounding boxes on the images.
- OpenCV: For basic image processing (can be extended).
- NumPy: For handling arrays and image data preprocessing.
- os: To manage file paths and directories.

---

How It Works
1. Image Upload: The user uploads a plant leaf image through the web interface.
2. Image Processing: The image is processed and resized to match the model’s input dimensions.
3. Disease Prediction: The pre-trained CNN model predicts the disease from the uploaded image.
4. Display Results: The system shows the predicted disease and its associated remedy, along with the image showing a bounding box around the affected area.
5. Feedback Loop: Users can upload another image or provide feedback if necessary.

---

Installation

 1. Clone the Repository
```bash
git clone https://github.com/your-repo-url/plant-disease-detection
cd plant-disease-detection
```

 2. Set up Virtual Environment (Optional but recommended)
```bash
python -m venv venv
source venv/bin/activate   On Windows use `venv\Scripts\activate`
```

 3. Install Required Dependencies
Install the necessary Python libraries:
```bash
pip install -r requirements.txt
```

 4. Download the Pre-trained Model
- Place the trained `plant_disease_model.h5` file in the root directory of the project. This model will be used for predictions.
- Make sure the model is compatible with TensorFlow 2.x and uses image input size (e.g., 64x64) matching the preprocessing done in the project.

 5. Run the Flask Application
```bash
python app.py
```

- The app will be hosted locally on `http://127.0.0.1:5000/`. Open this link in your browser.

---

Usage

1. Open the web browser and navigate to `http://127.0.0.1:5000/`.
2. Upload an image of a plant leaf.
3. The system will process the image and display the result, including:
   - The detected plant disease or indication if the plant is healthy.
   - A remedy or treatment recommendation for the detected disease.
4. You can upload another image to continue detecting plant diseases.

---

Project Structure

```bash
plant-disease-detection/
├── static/
│   └── result_image.png    Folder to store output images
├── templates/
│   ├── index.html          Frontend for uploading image
│   └── result.html         Frontend for displaying results
├── app.py                  Main Flask app
├── plant_disease_model.h5   Trained CNN model (not included in repo)
├── requirements.txt        Python dependencies
└── README.md               Project documentation (this file)
```

---

Key Python Libraries

- Flask: Manages the web framework, routing, and request handling.
- TensorFlow/Keras: Loads and runs the trained CNN model for plant disease detection.
- PIL (Pillow): Handles image processing, including resizing and drawing bounding boxes.
- OpenCV (cv2): Additional image processing, reading images, and converting color spaces.
- NumPy: Manipulates image arrays for preprocessing and input to the CNN model.
- os: Handles file operations, such as saving and loading images.

---

Model Details
- The Convolutional Neural Network (CNN) model is trained on a dataset of plant leaf images representing various diseases (e.g., Apple scab, Black rot, etc.) and healthy leaves.
- The model input size is 64x64 pixels.
- The trained model classifies images into categories such as diseases or healthy states, based on patterns it has learned from training data.

---

Contributing

If you'd like to contribute, please fork the repository, create a new branch, and submit a pull request with your changes. Make sure to follow coding standards and include appropriate comments in your code.

---

Acknowledgments
- TensorFlow and Keras teams for providing an easy-to-use deep learning framework.
- OpenCV and PIL (Pillow) for making image processing simple in Python.
- Flask for being a lightweight and flexible web framework.

---

This `README.md` provides an overview of the project and should help users understand the functionality, installation process, and usage instructions for the Plant Disease Detection System. Feel free to customize this file further as needed.