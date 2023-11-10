from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io

app = Flask(__name__)

# Load models
model_tomato = load_model('model.h5')
model_potato = load_model('potatoes.h5')

# Class labels for tomato and potato models
tomato_class_labels = ["Tomato - Bacteria Spot Disease", "Tomato - Early Blight Disease", "Tomato - Healthy and Fresh",
                       "Tomato - Late Blight Disease", "Tomato - Leaf Mold Disease", "Tomato - Septoria Leaf Spot Disease",
                       "Tomato - Target Spot Disease", "Tomato - Tomoato Yellow Leaf Curl Virus Disease",
                       "Tomato - Tomato Mosaic Virus Disease", "Tomato - Two Spotted Spider Mite Disease"]

potato_class_labels = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def predict_plant_tomato(image_base64, model, class_labels):
    img = image.load_img(io.BytesIO(image_base64), target_size=(128, 128))
    img_array = image.img_to_array(img) / 255
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    pred = np.argmax(result, axis=1)
    predicted_class = class_labels[int(pred)]
    probability = float(result[0, int(pred)])  # Extract the probability for the predicted class
    return predicted_class, probability

def predict_plant_potato(image_base64, model, class_labels):
    img = image.load_img(io.BytesIO(image_base64), target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    predicted_class = class_labels[np.argmax(result)]
    probability = float(np.max(result))
    return predicted_class, probability

def interpret_prediction(prediction, class_labels):
    max_index = np.argmax(prediction)
    max_class = class_labels[max_index]
    max_probability = float(prediction[max_index])
    interpreted_result = {"class": max_class, "probability": max_probability}
    return interpreted_result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_tomato', methods=['POST'])
def predict_tomato():
    if request.method == 'POST':
        image_file = request.files['image']
        
        # Check if the file is provided and has an allowed extension
        if image_file and allowed_file(image_file.filename):
            # Read the file content as bytes
            image_base64 = image_file.read()
            
            # Predict using the model
            prediction, probability = predict_plant_tomato(image_base64, model_tomato, tomato_class_labels)
            
            # Return the result
            interpreted_result = {"class": prediction, "probability": probability}
            return jsonify(interpreted_result)
        
        return jsonify({"error": "Invalid file"}), 400

    return jsonify({"error": "Method not allowed"}), 405

@app.route('/predict_potato', methods=['POST'])
def predict_potato():
    if request.method == 'POST':
        image_file = request.files['image']
        
        # Check if the file is provided and has an allowed extension
        if image_file and allowed_file(image_file.filename):
            # Read the file content as bytes
            image_base64 = image_file.read()
            
            # Predict using the model
            prediction, probability = predict_plant_potato(image_base64, model_potato, potato_class_labels)
            
            # Return the result
            interpreted_result = {"class": prediction, "probability": probability}
            return jsonify(interpreted_result)
        
        return jsonify({"error": "Invalid file"}), 400

    return jsonify({"error": "Method not allowed"}), 405

# Define a function to check if the file has an allowed extension
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run()
