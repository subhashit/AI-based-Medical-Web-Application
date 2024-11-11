from flask import Flask, request, render_template, flash, redirect, url_for

from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime


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
    # One-to-many relationship with MedicalRecord
    medical_records = db.relationship('MedicalRecord', backref='owner', lazy=True)
    # Set encrypted password
    def set_password(self, password):
        self.password = bcrypt.generate_password_hash(password).decode('utf-8')
    # Check if the entered password is correct
    def check_password(self, password):
        return bcrypt.check_password_hash(self.password, password)
    
class MedicalRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    record_date = db.Column(db.Date, nullable=False)
    blood_pressure = db.Column(db.String(100))
    pulse = db.Column(db.Integer)
    glucose = db.Column(db.Integer)
    cholesterol_levels = db.Column(db.Integer)
    heart_rate = db.Column(db.Integer)
    oxygen_saturation = db.Column(db.Integer)
    respiratory_rate = db.Column(db.Integer)
    liver_function_test = db.Column(db.String(100))
    kidney_function_test = db.Column(db.String(100))
    blood_report = db.Column(db.String(100))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    user = db.relationship('User', backref=db.backref('records', lazy=True))
    
# Load user for login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
# Create the database
with app.app_context():
    db.create_all()
# **database part end**

# ***image analysis part start***
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

@app.route('/index', methods = ['GET'])
@login_required
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
@login_required
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
@login_required  # Ensure the user is logged in
def dashboard():
    return render_template('dashboard.html', user=current_user)


# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

# **inside dashboard**
# Route for User Profile

@app.route('/user_profile', methods=['GET', 'POST'])
@login_required
def user_profile():
    if request.method == 'POST':
        # Get form data from POST request
        blood_pressure = request.form['blood_pressure']
        pulse = request.form['pulse']
        glucose = request.form['glucose']
        cholesterol_levels = request.form['cholesterol_levels']
        heart_rate = request.form['heart_rate']
        oxygen_saturation = request.form['oxygen_saturation']
        respiratory_rate = request.form['respiratory_rate']
        liver_function_test = request.form['liver_function_test']
        kidney_function_test = request.form['kidney_function_test']
        blood_report = request.form['blood_report']
        record_date = request.form['record_date']

        # Parse the record_date string to a datetime object
        record_date = datetime.strptime(record_date, '%Y-%m-%d')

        # Create a new medical record
        new_record = MedicalRecord(
            user_id=current_user.id,  # Reference the current logged-in user
            blood_pressure=blood_pressure,
            pulse=pulse,
            glucose=glucose,
            cholesterol_levels=cholesterol_levels,
            heart_rate=heart_rate,
            oxygen_saturation=oxygen_saturation,
            respiratory_rate=respiratory_rate,
            liver_function_test=liver_function_test,
            kidney_function_test=kidney_function_test,
            blood_report=blood_report,
            record_date=record_date
        )

        # Add and commit the new record
        db.session.add(new_record)
        db.session.commit()

        return redirect(url_for('user_profile'))

    # Query all medical records for the current logged-in user
    records = MedicalRecord.query.filter_by(user_id=current_user.id).all()
    return render_template('user_profile.html', records=records)
# Route to handle record deletion
@app.route('/delete_record/<int:record_id>', methods=['POST'])
@login_required
def delete_record(record_id):
    record = MedicalRecord.query.get(record_id)
    if record and record.user_id == current_user.id:
        db.session.delete(record)
        db.session.commit()
        flash('Record deleted successfully!', 'success')
    else:
        flash('Record not found or unauthorized action!', 'danger')
    return redirect(url_for('user_profile'))


# Route for Disease Prediction
@app.route('/disease_prediction')
@login_required
def disease_prediction():
    return render_template('disease_prediction.html')

# Route for Medical Chatbot
@app.route('/medical_chatbot')
@login_required
def medical_chatbot():
    return render_template('medical_chatbot.html')
# ** dashboard end**



if __name__ == '__main__':
    app.run(debug=True)
