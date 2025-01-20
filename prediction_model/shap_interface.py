import json
from .neural_network import CropPredictor
from .shap_utils import SHAPUtility

# Initialize components
predictor = CropPredictor()
explainer = SHAPUtility.initialize_shap_explainer(predictor)

def predict_shap_values(request_body):
    """
    Compute SHAP values for the given input data.
    """
    try:
        # Parse input data
        data = json.loads(request_body)
        required_params = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

        # Validate input
        for param in required_params:
            if param not in data:
                return {"error": f"Missing required parameter: {param}"}

        # Extract input features
        X = [[data["N"], data["P"], data["K"], data["temperature"], 
              data["humidity"], data["ph"], data["rainfall"]]]

        # Compute SHAP values
        shap_values = SHAPUtility.get_shap_values(X, predictor, explainer)

        return {"shap_values": shap_values.tolist()}

    except Exception as e:
        return {"error": str(e)}

def generate_shap_diagram(request_body):
    """
    Generate a SHAP diagram (waterfall or force plot).
    """
    try:
        # Parse input data
        data = json.loads(request_body)
        required_params = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "plot_type"]

        # Validate input
        for param in required_params:
            if param not in data:
                return {"error": f"Missing required parameter: {param}"}

        # Extract input features
        X = [[data["N"], data["P"], data["K"], data["temperature"], 
              data["humidity"], data["ph"], data["rainfall"]]]
        plot_type = data["plot_type"]

        # Compute SHAP values and generate diagram
        # shap_values = SHAPUtility.get_shap_values(X, predictor, explainer)
        # diagram_base64 = SHAPUtility.get_shap_diagram(X, predictor, explainer, shap_values, plot_type)
        diagram_base64 = SHAPUtility.generate_shap_diagram(data)
        return {"shap_diagram": diagram_base64}

    except Exception as e:
        return {"error": str(e)}

def generate_global_shap_summary(request_body):
    """
    Generate a global SHAP summary plot for the model.
    """
    try:
        # Parse input data
        data = json.loads(request_body)
        num_samples = data.get("num_samples", 100)

        # Generate global summary
        global_summary_base64 = SHAPUtility.get_shap_global_explanation(predictor, explainer, num_samples)

        return {"global_summary": global_summary_base64}

    except Exception as e:
        return {"error": str(e)}
