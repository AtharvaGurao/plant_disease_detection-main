import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageDraw

# Initialize Flask app
app = Flask(__name__)

# Load the trained plant disease model
model = load_model('plant_disease_model.h5')

# Define disease class names (update these according to your model's output classes)
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
               'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                     'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                       'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                         'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
                           'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                           'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
                             'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
                             'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
                             'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']  # Replace with your actual class names

remedies = {
    'Apple___Apple_scab': 'Apply fungicides such as captan or myclobutanil. Prune affected leaves and ensure proper air circulation.',
    'Apple___Black_rot': 'Remove and destroy infected fruits and branches. Use fungicides during growing season.',
    'Apple___Cedar_apple_rust': 'Use resistant apple varieties. Apply fungicide sprays in the spring.',
    'Apple___healthy': 'No action required. Continue proper care.',
    'Blueberry___healthy': 'No action required. Continue proper care.',
    'Cherry_(including_sour)___Powdery_mildew': 'Apply fungicide at first sign of infection. Remove affected plant parts.',
    'Cherry_(including_sour)___healthy': 'No action required. Continue proper care.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Use resistant hybrids. Apply fungicide at the first sign of disease.',
    'Corn_(maize)___Common_rust_': 'Use resistant varieties. Apply fungicide if needed.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Use disease-resistant seeds. Apply fungicides.',
    'Corn_(maize)___healthy': 'No action required. Continue proper care.',
    'Grape___Black_rot': 'Prune affected areas. Apply fungicides and ensure good air circulation.',
    'Grape___Esca_(Black_Measles)': 'Remove and destroy affected vines. Improve drainage and avoid injuries to the vines.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Use fungicides and ensure good air circulation. Remove affected leaves.',
    'Grape___healthy': 'No action required. Continue proper care.',
    'Orange___Haunglongbing_(Citrus_greening)': 'No known cure. Remove and destroy infected trees. Control insect vector (psyllids).',
    'Peach___Bacterial_spot': 'Apply copper-based fungicides. Use disease-resistant varieties.',
    'Peach___healthy': 'No action required. Continue proper care.',
    'Pepper,_bell___Bacterial_spot': 'Apply copper fungicides. Remove and destroy affected plants.',
    'Pepper,_bell___healthy': 'No action required. Continue proper care.',
    'Potato___Early_blight': 'Use disease-free seed and apply fungicides regularly.',
    'Potato___Late_blight': 'Apply fungicides and destroy infected plants. Use resistant varieties.',
    'Potato___healthy': 'No action required. Continue proper care.',
    'Raspberry___healthy': 'No action required. Continue proper care.',
    'Soybean___healthy': 'No action required. Continue proper care.',
    'Squash___Powdery_mildew': 'Apply sulfur-based fungicides. Remove affected leaves.',
    'Strawberry___Leaf_scorch': 'Prune affected leaves. Apply fungicides as necessary.',
    'Strawberry___healthy': 'No action required. Continue proper care.',
    'Tomato___Bacterial_spot': 'Remove and destroy affected plants. Apply copper-based fungicides.',
    'Tomato___Early_blight': 'Apply fungicides and rotate crops. Remove affected plant debris.',
    'Tomato___Late_blight': 'Use resistant varieties and apply fungicides. Remove affected plants.',
    'Tomato___Leaf_Mold': 'Ensure good air circulation. Apply fungicides.',
    'Tomato___Septoria_leaf_spot': 'Apply fungicides and remove affected leaves. Avoid overhead watering.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Apply miticides. Spray water to dislodge mites from leaves.',
    'Tomato___Target_Spot': 'Remove affected leaves and apply fungicides.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Control whitefly, the insect vector. Remove and destroy affected plants.',
    'Tomato___Tomato_mosaic_virus': 'Remove and destroy infected plants. Disinfect tools and use resistant varieties.',
    'Tomato___healthy': 'No action required. Continue proper care.'
}


# Function to predict the disease and draw bounding box
def predict_and_draw_bounding_box(img_path):
    img = Image.open(img_path)
    
    # Resize the image to match the input size of the model (64x64 in our case)
    img_resized = img.resize((64, 64))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_names[predicted_class]

    # Draw bounding box (For illustration, drawing a fixed box. You can use object detection techniques to improve this)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(50, 50), (150, 150)], outline="red", width=3)  # Example static box

    # Save the modified image with bounding box
    result_image_path = os.path.join('static', 'result_image.png')
    img.save(result_image_path)
    
    return predicted_label, result_image_path

# Home route for the upload form
@app.route('/')
def upload_form():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        file_path = os.path.join('static', file.filename)
        file.save(file_path)
        
        # Predict disease and draw bounding box
        predicted_label, result_image_path = predict_and_draw_bounding_box(file_path)
        
        # Get remedy for the predicted disease
        remedy = remedies.get(predicted_label, 'No remedy available')

        # Render result.html with the prediction, image, and remedy
        return render_template('result.html', label=predicted_label, image_path=result_image_path, remedy=remedy)


if __name__ == "__main__":
    app.run(debug=True)
