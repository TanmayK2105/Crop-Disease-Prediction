from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models for different plants
model_tomato = load_model('tomato.h5')
model_potato = load_model('potatoes.h5')
model_apple = load_model('apple.h5')
model_grapes = load_model('grapes_disease_model_with_early_stopping.h5')
# model_oranges = load_model('model_oranges.h5')
# model_soybean = load_model('model_soybean.h5')
# model_strawberry = load_model('model_strawberry.h5')
# model_peach = load_model('model_peach.h5')
model_corn = load_model('corn.h5')

# Class labels for different plants
tomato_class_labels = ["Tomato - Bacteria Spot Disease", "Tomato - Early Blight Disease", "Tomato - Healthy", "Tomato - Late Blight Disease", "Tomato - Leaf Mold Disease", "Tomato - Septoria Leaf Spot Disease", "Tomato - Target Spot Disease", "Tomato - Yellow Leaf Curl Virus Disease", "Tomato - Tomato Mosaic Virus Disease", "Tomato - Two Spotted Spider Mite Disease"]

potato_class_labels = ['Potato - Early Blight', 'Potato - Late Blight', 'Potato - Healthy']

apple_class_labels = ['Apple : Scab',
 'Apple : Black Rot',
 'Apple : Cedar rust',
 'Apple : Healthy']

grapes_class_labels = ['Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy']

oranges_class_labels = ['Orange - HLB (Citrus Greening)', 'Orange - Healthy']

soybean_class_labels = ['Soybean - Healthy', 'Soybean - Mosaic Virus']

strawberry_class_labels = ['Strawberry - Healthy', 'Strawberry - Leaf Scorch']

peach_class_labels = ['Peach - Bacterial Spot', 'Peach - Healthy']

corn_class_labels = ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy']

def predict_tomato(image_bytes, model, class_labels):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((128, 128))  # Resize the image to match the model's expected sizing
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    pred = np.argmax(result, axis=1)
    predicted_class = class_labels[int(pred)]
    probability = float(result[0, int(pred)])  # Extract the probability for the predicted class
    return predicted_class, probability

def predict_plant(image_bytes, model, class_labels):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((256, 256))  # Resize the image to match the model's expected sizing
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    pred = np.argmax(result, axis=1)
    predicted_class = class_labels[int(pred)]
    probability = float(result[0, int(pred)])  # Extract the probability for the predicted class
    return predicted_class, probability

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return render_template('error.html', message="Error: No image file provided.")

        crop_type = request.form['crop_type']
        image_base64 = request.files['image'].read()

        if crop_type == 'tomato':
            prediction, _ = predict_tomato(image_base64, model_tomato, tomato_class_labels)
            return render_template('predict.html', crop_type='Tomato', prediction=prediction)
        elif crop_type == 'potato':
            prediction, _ = predict_plant(image_base64, model_potato, potato_class_labels)
            return render_template('predict.html', crop_type='Potato', prediction=prediction)
        elif crop_type == 'apple':
            prediction, _ = predict_plant(image_base64, model_apple, apple_class_labels)
            return render_template('predict.html', crop_type='Apple', prediction=prediction)
        elif crop_type == 'grapes':
            prediction, _ = predict_plant(image_base64, model_grapes, grapes_class_labels)
            return render_template('predict.html', crop_type='Grapes', prediction=prediction)
        elif crop_type == 'oranges':
            prediction, _ = predict_plant(image_base64, model_oranges, oranges_class_labels)
            return render_template('predict.html', crop_type='Oranges', prediction=prediction)
        elif crop_type == 'soybean':
            prediction, _ = predict_plant(image_base64, model_soybean, soybean_class_labels)
            return render_template('predict.html', crop_type='Soybean', prediction=prediction)
        elif crop_type == 'strawberry':
            prediction, _ = predict_plant(image_base64, model_strawberry, strawberry_class_labels)
            return render_template('predict.html', crop_type='Strawberry', prediction=prediction)
        elif crop_type == 'peach':
            prediction, _ = predict_plant(image_base64, model_peach, peach_class_labels)
            return render_template('predict.html', crop_type='Peach', prediction=prediction)
        elif crop_type == 'corn':
            prediction, _ = predict_plant(image_base64, model_corn, corn_class_labels)
            return render_template('predict.html', crop_type='Corn', prediction=prediction)
        else:
            return render_template('error.html', message="Error: Invalid crop type.")

    except Exception as e:
        print(f"Error processing prediction: {str(e)}")
        return render_template('error.html', message=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run()
