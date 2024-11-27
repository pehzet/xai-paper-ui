from flask import Flask, request, jsonify
from .neuronal_network import CropPredictor
import json
USE_FLASK = False
# Initialize the Flask app
if USE_FLASK:
    app = Flask(__name__)

# Initialize the Crop Predictor model
predictor = CropPredictor()

# Load the function configuration
with open('function_config.json', 'r') as config_file:
    function_config = json.load(config_file)

# @app.route('/predict', methods=['POST'])
def predict(data):
    """
    Endpoint to predict crop based on soil and environmental characteristics.
    """
    try:
        # Parse request data
        # data = request.get_json()
        data = json.loads(data)

        # Validate required parameters
        required_params = function_config[0]['function']['parameters']['required']
        missing_params = [param for param in required_params if param not in data]
        if missing_params:
            return jsonify({"error": f"Missing required parameters: {', '.join(missing_params)}"}), 400

        # Extract parameters
        N = data.get("N")
        P = data.get("P")
        K = data.get("K")
        temperature = data.get("temperature")
        humidity = data.get("humidity")
        ph = data.get("ph")
        rainfall = data.get("rainfall")
        
        # Make prediction
        prediction = predictor.predict(N, P, K, temperature, humidity, ph, rainfall)
        print(f"Predicted crop: {prediction[0]}")
        if USE_FLASK:
            # Respond with prediction
            return jsonify({"predicted_crop": prediction[0]}), 200
        else:
            return prediction[0]
    except Exception as e:
        if USE_FLASK:
            return jsonify({"error": str(e)}), 500
        else:
            return {"error": str(e)}
        
if USE_FLASK and __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)
