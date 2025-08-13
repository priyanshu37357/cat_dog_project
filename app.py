from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Model load karo (make sure cat_dog_model.h5 same folder me ho)
model = load_model("cat_dog_model.h5")

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# About page
@app.route('/about')
def about():
    return render_template('about.html')

# Contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Prediction page
@app.route('/project', methods=['GET', 'POST'])
def project():
    prediction = None
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('project.html', prediction="No file uploaded!")

        file = request.files['image']

        if file.filename == '':
            return render_template('project.html', prediction="No file selected!")

        # Save uploaded file
        filepath = os.path.join('static', file.filename)
        file.save(filepath)

        # Image preprocessing
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        pred = model.predict(img_array)
        if pred[0][0] > 0.5:
            prediction = "Dog 🐶"
        else:
            prediction = "Cat 🐱"

        return render_template('project.html', prediction=prediction)

    return render_template('project.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
