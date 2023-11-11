from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import io
import os

app = Flask(__name__)

# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load models using absolute paths
model_tomato_path = os.path.join(script_dir, 'model.h5')
model_potato_path = os.path.join(script_dir, 'potatoes.h5')

# Check if model files exist
if not os.path.exists(model_tomato_path):
    raise FileNotFoundError(f"Model file not found: {model_tomato_path}")

if not os.path.exists(model_potato_path):
    raise FileNotFoundError(f"Model file not found: {model_potato_path}")

# Load models
model_tomato = load_model(model_tomato_path)
model_potato = load_model(model_potato_path)

# Class labels for tomato and potato models
tomato_class_labels = ["Tomato - Bacteria Spot Disease", "Tomato - Early Blight Disease", "Tomato - Healthy and Fresh",
                       "Tomato - Late Blight Disease", "Tomato - Leaf Mold Disease", "Tomato - Septoria Leaf Spot Disease",
                       "Tomato - Target Spot Disease", "Tomato - Tomato Yellow Leaf Curl Virus Disease",
                       "Tomato - Tomato Mosaic Virus Disease", "Tomato - Two Spotted Spider Mite Disease"]

potato_class_labels = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def predict_plant_tomato(image_bytes, model, class_labels):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    pred = np.argmax(result, axis=1)
    predicted_class = class_labels[int(pred)]
    probability = float(result[0, int(pred)])
    return predicted_class, probability

def predict_plant_potato(image_bytes, model, class_labels):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    pred = np.argmax(result, axis=1)
    predicted_class = class_labels[int(pred)]
    probability = float(result[0, int(pred)])
    return predicted_class, probability

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_tomato', methods=['POST'])
def predict_tomato():
    try:
        if 'image' not in request.files:
            return "Error: No image file provided."

        image_base64 = request.files['image'].read()
        prediction, _ = predict_plant_tomato(image_base64, model_tomato, tomato_class_labels)
        return f"Predicted Class: {prediction}"

    except Exception as e:
        print(f"Error processing tomato prediction: {str(e)}")
        return f"Error: {str(e)}"

@app.route('/predict_potato', methods=['POST'])
def predict_potato():
    try:
        if 'image' not in request.files:
            return "Error: No image file provided."

        image_base64 = request.files['image'].read()
        prediction, _ = predict_plant_potato(image_base64, model_potato, potato_class_labels)
        return f"Predicted Class: {prediction}"

    except Exception as e:
        print(f"Error processing potato prediction: {str(e)}")
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run()
