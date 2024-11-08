import os
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
from keras.models import load_model
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template

# Initialize the Flask app
app = Flask(__name__)

model = load_model('BrainTumor.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"

def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64,64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)

    predicition = model.predict(input_img)[0][0]

    result = 1 if predicition >= 0.5 else 0
    return result

@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)

        value = getResult(file_path)
        result = get_className(value) 
        return result
    
    return render_template('index.html')

# Home Route
@app.route('/')
def home():
    return render_template('home.html')

# Login Route
@app.route('/login')
def login():
    return render_template('login.html')

# Register Route
@app.route('/register')
def register():
    return render_template('register.html')

if __name__ == '__main__':
    app.run(debug=True)
