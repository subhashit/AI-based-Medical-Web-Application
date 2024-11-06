from flask import Flask, render_template

# Initialize the Flask app
app = Flask(__name__)

# Home Route
@app.route('/')
def home():
    return render_template('home.html')

# Login Route
@app.route('/login')
def login():
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)
