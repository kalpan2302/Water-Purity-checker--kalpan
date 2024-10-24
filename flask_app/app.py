from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)

# Load the models
model_original = load_model("models/water_purity_model.h5")
model_vgg16 = load_model("models/water_purity_model_vgg16.h5")

# Prediction function
def predict_image(image_path, model):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return 'Dirty' if prediction > 0.5 else 'Clean'

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle file upload
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Ensure the directory exists
            images_dir = os.path.join('static', 'images')
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)

            # Save the file
            filename = file.filename
            filepath = os.path.join(images_dir, filename)
            file.save(filepath)
            
            selected_model = request.form.get('model')
            result_original, result_vgg16, comparison = None, None, None
            
            if selected_model == "Original Model":
                result_original = predict_image(filepath, model_original)
            elif selected_model == "VGG16 Model":
                result_vgg16 = predict_image(filepath, model_vgg16)
            elif selected_model == "Compare Both":
                result_original = predict_image(filepath, model_original)
                result_vgg16 = predict_image(filepath, model_vgg16)
                comparison = "Match" if result_original == result_vgg16 else "Mismatch"
            
            return render_template('index.html', image=filename, 
                                   result_original=result_original, 
                                   result_vgg16=result_vgg16, 
                                   comparison=comparison)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)

# Load the models
model_original = load_model("models/water_purity_model.h5")
model_vgg16 = load_model("models/water_purity_model_vgg16.h5")

# Prediction function
def predict_image(image_path, model):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return 'Dirty' if prediction > 0.5 else 'Clean'

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle file upload
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Ensure the directory exists
            images_dir = os.path.join('static', 'images')
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)

            # Save the file
            filename = file.filename
            filepath = os.path.join(images_dir, filename)
            file.save(filepath)
            
            selected_model = request.form.get('model')
            result_original, result_vgg16, comparison = None, None, None
            
            if selected_model == "Original Model":
                result_original = predict_image(filepath, model_original)
            elif selected_model == "VGG16 Model":
                result_vgg16 = predict_image(filepath, model_vgg16)
            elif selected_model == "Compare Both":
                result_original = predict_image(filepath, model_original)
                result_vgg16 = predict_image(filepath, model_vgg16)
                comparison = "Match" if result_original == result_vgg16 else "Mismatch"
            
            return render_template('index.html', image=filename, 
                                   result_original=result_original, 
                                   result_vgg16=result_vgg16, 
                                   comparison=comparison)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
