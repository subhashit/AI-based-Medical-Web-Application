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

import joblib

#////////////////////////////////////////////////////////////////////////////
# Initialize the Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'medic'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
#//////////////////////////////////////////////////////////////////////////


#////////////////////////////////////////////////////////////////////////////////////////start
#DATABASE
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
#/////////////////////////////////////////////////////////////////////////////////end





#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////start
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
#///////////////////////////////////////////////////////////////////////////////////////////end




# Home Route
@app.route('/')
def home():
    return render_template('home.html')



#/////////////////////////////////////////////////////////////////////////////////start
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

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))
#//////////////////////////////////////////////////////////////////////////////end




#//////////////////////////////////////////////////////////////////////////////////start
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
#/////////////////////////////////////////////////////////////////////////////////////////end





#////////////////////////////////////////////////////////////////////////////////////////start
# Dashboard page route (protected by login_required)
@app.route('/dashboard')
@login_required  # Ensure the user is logged in
def dashboard():
    return render_template('dashboard.html', user=current_user)

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
#/////////////////////////////////////////////////////////////////////////////////////////end








#////////////////////////////////////////////////////////////////////////////////////////////////////////////start
#  Disease_prediction
models = {
    "diabetes": joblib.load("models/diabetes.pkl"),
    "liver": joblib.load("models/liver.pkl"),
    "heart": joblib.load("models/heart.pkl")
}
@app.route("/disease_prediction", methods=["GET", "POST"])
@login_required
def disease_prediction():
    result = None
    message = None
    error = None

    if request.method == "POST":
        # Get the selected model
        selected_model = request.form.get("model")
        if selected_model not in models:
            error = "Invalid model selected."
        else:
            # Collect input data based on the selected model
            try:
                if selected_model == "diabetes":
                    fields = [
                        "Pregnancies", "Glucose", "BloodPressure",
                        "SkinThickness", "Insulin", "BMI",
                        "DiabetesPedigreeFunction", "Age"
                    ]
                elif selected_model == "liver":
                    fields = [
                        "Age", "Gender", "TB", "DB", "Alkphos",
                        "Sgpt", "Sgot", "TP", "ALB", "A/G Ratio"
                    ]
                elif selected_model == "heart":
                    fields = [
                        "Age", "Sex", "Chest pain type", "BP", "Cholesterol",
                        "FBS over 120", "EKG results", "Max HR", "Exercise angina",
                        "ST depression", "Slope of ST", "Number of vessels fluro", "Thallium"
                    ]

                # Extract and process form data
                input_data = []
                for field in fields:
                    value = request.form.get(field, "").strip()
                    if field in ["Gender", "Sex", "Exercise angina"]:  # Handle categorical fields
                        if field in ["Gender", "Sex"]:
                            value = 1 if value.lower() == "male" else 0
                        elif field == "Exercise angina":
                            value = 1 if value.lower() == "yes" else 0
                    input_data.append(float(value))

                # Make prediction
                model = models[selected_model]
                prediction = model.predict([input_data])[0]  # Get prediction result (0 or 1)

                # Set the message based on the prediction
                if prediction == 0:
                    result = f"You are not suffering from {selected_model.capitalize()}."
                elif prediction == 1:
                    result = f"You are suffering from {selected_model.capitalize()}."
            except Exception as e:
                error = f"Error in processing input: {e}"

    return render_template(
        "disease_prediction.html",
        result=result,
        error=error
    )
#/////////////////////////////////////////////////////////////////////////////////////////////////end



# Route for Medical Chatbot
@app.route('/medical_chatbot')
@login_required
def medical_chatbot():
    return render_template('medical_chatbot.html')
# ** dashboard end**

if __name__ == '__main__':
    app.run(debug=True)
