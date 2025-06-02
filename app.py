# app.py
from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import logging
import os
from flask_cors import CORS  # To allow cross-origin requests from frontend

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load model and scaler
model = joblib.load("best_wine_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define expected input features
expected_features = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline"
]

@app.route('/')
def home():
    return "🍷 Wine Prediction API is up and running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
    except Exception as e:
        logging.warning(f"Invalid JSON received: {str(e)}")
        return jsonify({"error": "Invalid JSON data"}), 400

    logging.info(f"Received request data: {data}")

    # Check for missing features
    missing = [feature for feature in expected_features if feature not in data]
    if missing:
        msg = f"Missing input features: {', '.join(missing)}"
        logging.warning(msg)
        return jsonify({"error": msg}), 400

    try:
        # Create input array in correct order
        features = np.array([[data[feat] for feat in expected_features]])

        # Scale features and make prediction
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)

        result = {'predicted_class': int(prediction[0])}
        logging.info(f"Prediction result: {result}")
        return jsonify(result)

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Prediction failed. Check your input values."}), 500

@app.route('/form')
def form():
    # Serve the HTML form - make sure index.html is in the same directory as app.py
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/styles.css')
def styles():
    # Serve the CSS file if needed
    return send_from_directory(os.getcwd(), 'styles.css')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
