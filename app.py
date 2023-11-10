from flask import Flask, request, render_template
import numpy as np
from PIL import Image 
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

def predict_plant_tomato(image_bytes, model, class_labels):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((128, 128))  # Resize the image to match the model's expected sizing
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
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

@app.route('/predict_tomato', methods=['GET', 'POST'])
def predict_tomato():
    prediction = None
    if request.method == 'POST':
        image_base64 = request.files['image'].read()
        print("Received image_base64:", image_base64[:50])  # Print the first 50 characters for debugging
        prediction, probability = predict_plant_tomato(image_base64, model_tomato, tomato_class_labels)
        print("Prediction:", prediction)
        print("Probability:", probability)
    return render_template('predict_tomato.html', prediction=prediction)


@app.route('/predict_potato', methods=['GET', 'POST'])
def predict_potato():
    prediction = None
    if request.method == 'POST':
        image_base64 = request.files['image'].read()
        print("Received image_base64:", image_base64[:50])  # Print the first 50 characters for debugging
        prediction, probability = predict_plant_potato(image_base64, model_potato, potato_class_labels)
        print("Prediction:", prediction)
        print("Probability:", probability)
    return render_template('predict_potato.html', prediction=prediction)


if __name__ == '__main__':
    app.run()
