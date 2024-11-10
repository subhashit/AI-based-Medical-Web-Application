from flask import Flask, request, render_template, flash, redirect, url_for

from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user


import os
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
from keras.models import load_model
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'medic'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

# **database part start**
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
# User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    username = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    # Set encrypted password
    def set_password(self, password):
        self.password = bcrypt.generate_password_hash(password).decode('utf-8')
    # Check if the entered password is correct
    def check_password(self, password):
        return bcrypt.check_password_hash(self.password, password)
# Load user for login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
# Create the database
with app.app_context():
    db.create_all()
# **database part end**

# ***image analysis part start***
model = load_model(r'C:\Users\subha\OneDrive\Desktop\project\AI-based-Medical-Web-Application\medical_app\BrainTumor.h5')
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

@app.route('/index', methods = ['GET'])
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
# ***image analysis part end ***

# Home Route
@app.route('/')
def home():
    return render_template('home.html')

# Login page route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Login Failed. Check username and/or password.', 'danger')
    return render_template('login.html')

# Registration page route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))

        # Check if the username already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'danger')
            return redirect( url_for('register'))

        # Create a new user and hash the password
        new_user = User(name=name, username=username)
        new_user.set_password(password)

        # Add the new user to the database
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# Dashboard page route (protected by login_required)
@app.route('/dashboard')
@login_required
def dashboard():
    return f'Welcome {current_user.username}! This is your dashboard.'

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
