# app.py
from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import logging
import os

# Initialize Flask app
app = Flask(__name__)

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
    data = request.get_json(force=True)
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
        return jsonify({"error": str(e)}), 500

@app.route('/form')
def form():
    # Serve HTML form file (should be placed in the same directory as this script)
    return send_from_directory(os.getcwd(), 'index.html')

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)

