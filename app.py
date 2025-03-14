from flask import Flask, render_template, request, redirect, session, url_for, flash
import os
import cv2
import numpy as np
import tensorflow as tf
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

# Initialize Flask app and Flask-Login
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = 'your_secret_key'  # Required for session management

# Define the upload folder path
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # This folder should exist in your project

# Set up Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Redirect to 'login' if not logged in

# Dummy user data for the purpose of this example
users = {'admin': {'password': 'adminpass'}}

# Class for User
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# Load the user for session
@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Function to load and preprocess a single image
def load_and_preprocess_image(image_path, image_size=(128, 128)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, image_size)
    image = np.array(image, dtype="float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict if the image contains a tumor or not
def predict_image(image_path):
    model = tf.keras.models.load_model('model/brain_tumor_model.h5')
    preprocessed_image = load_and_preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_label = 1 if prediction > 0.5 else 0
    if predicted_label == 1:
        prediction_text = "Tumor detected"
    else:
        prediction_text = "No tumor detected"
    return image_path, prediction_text

# Route to the login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if user exists in the dummy users dictionary
        if username in users and users[username]['password'] == password:
            user = User(username)
            login_user(user)
            flash("Login Successful!", "success")
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password", "danger")
    
    return render_template('login.html')

# Route to the registration page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Check if password and confirm_password match
        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(request.url)
        
        # Check if username already exists
        if username in users:
            flash("Username already exists!", "danger")
            return redirect(request.url)
        
        # Add new user to the 'users' dictionary
        users[username] = {'password': password}
        flash("Registration successful! You can now login.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

# Route to logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

# Route to the home page
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            # Use the configured upload folder
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # Use the prediction function to make a prediction
            _, prediction_text = predict_image(image_path)
            
            # Pass the filename and prediction text to the template
            return render_template('index.html', prediction_text=prediction_text, image_path=filename)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)




