from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "🚗 Vehicle Detection API Running Successfully!"
