from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from PIL import Image

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model from this directory
model = keras.models.load_model(r"C:\Users\user\Downloads\image-classifier-main\image-classifier-main\custom_model.h5")

# Define a function to preprocess an image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def decode_predictions_custom(predictions):
    class_labels = {
        0: 'COVID-19 - This is a COVID-19 case with specific characteristics.',
        1: 'Non-COVID - This is not a COVID-19 case, but it might have other characteristics.',
        2: 'Normal - This is a normal case with no indications of COVID-19.'
    }
    max_class = np.argmax(predictions, axis=-1)
    decoded_labels = [class_labels[i] for i in max_class]
    return decoded_labels


# Define an API route for image classification
@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']

        if image_file.filename == '':
            return jsonify({'error': 'Empty file provided'}), 400

        if not image_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return jsonify({'error': 'Invalid file format'}), 400

        image_path = 'temp_image.jpg'
        image_file.save(image_path)

        img = preprocess_image(image_path)
        predictions = model.predict(img)

        # Use your custom decoding function
        class_labels = decode_predictions_custom(predictions)

        result = {'class': class_labels[0]}

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    images = request.files['images']
    image_path = os.path.join(app.root_path, 'images', images.filename)
    images.save(image_path)

    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)

    # Use your custom decoding function
    class_labels = decode_predictions_custom(yhat)

    classification = class_labels[0]

    return render_template('index.html', prediction=classification)

if __name__ == '__main__':
    app.run(port=3000, debug=True)

