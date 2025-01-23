import shap
import numpy as np
import matplotlib.pyplot as plt
import base64
import datetime
import pickle

import os
#from neuronal_network import CropPredictor
if __name__ == "__main__":
    from neural_network import CropPredictor
else:
    from .neural_network import CropPredictor

class SHAPUtility:
    """
    A utility class for handling SHAP explainers and generating explanations/visualizations.
    """
    @staticmethod
    def initialize_shap_explainer(predictor, num_background=100):
        """
        Initialize and return the SHAP explainer with valid background data.
        
        Parameters:
        - predictor (CropPredictor): The trained model predictor.
        - num_background (int): Number of background samples to use.
        
        Returns:
        - explainer (shap.KernelExplainer): The initialized SHAP explainer.
        """
        # Extract and preprocess training data for background
        X_train = predictor.X_train  # Already numerical and processed
        X_train_processed = predictor.preprocess_data(X_train, fit=False)
        
        # Select a subset of the training data for background
        background = X_train_processed[:num_background].astype(np.float32)

        # Define model prediction function for SHAP
        def model_predict(data):
            """
            Wrap the predictor's model prediction to match SHAP's requirements.
            
            Parameters:
            - data (numpy.ndarray): Input data for predictions.
            
            Returns:
            - predictions (numpy.ndarray): Class probabilities for each input.
            """
            # Ensure data is preprocessed in the same way as training data
            predictions = predictor.model.predict(data, verbose=0)
            return predictions

        # Initialize SHAP KernelExplainer
        explainer = shap.KernelExplainer(model_predict, background)
        return explainer

    @staticmethod
    def get_shap_values(X, predictor, explainer):
        """
        Get SHAP values for the given input data.
        """
        X_processed = predictor.preprocess_data(X, fit=False)
        shap_values = explainer.shap_values(X_processed)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        return np.squeeze(shap_values)

    @staticmethod
    def generate_shap_diagram(input_data):
        """
        Generate a SHAP diagram (waterfall or force plot) for a single sample based on input features.

        Parameters:
        - input_data (dict): JSON-like dictionary containing the input variables:
            - N (int): Ratio of Nitrogen content in the soil.
            - P (int): Ratio of Phosphorous content in the soil.
            - K (int): Ratio of Potassium content in the soil.
            - temperature (float): Temperature in degree Celsius.
            - humidity (float): Relative humidity in %.
            - ph (float): pH value of the soil.
            - rainfall (float): Rainfall in mm.
            - plot_type (str): Type of SHAP diagram ("waterfall" or "force").

        Returns:
        - plot_filename (str): The base64-encoded string of the SHAP plot image.
        """
        # Extract variables from input JSON using .get() with default None
        N = input_data.get("N", None)
        P = input_data.get("P", None)
        K = input_data.get("K", None)
        temperature = input_data.get("temperature", None)
        humidity = input_data.get("humidity", None)
        ph = input_data.get("ph", None)
        rainfall = input_data.get("rainfall", None)
        plot_type = input_data.get("plot_type", "waterfall")  # Default to "waterfall" if not provided

        # Validate plot_type
        if plot_type not in ["waterfall", "force"]:
            raise ValueError("Invalid plot_type. Choose 'waterfall' or 'force'.")

        # Initialize predictor and SHAP explainer
        predictor = CropPredictor()
        explainer = SHAPUtility.initialize_shap_explainer(predictor)

        # Format input data into expected shape
        X = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Predict the class
        class_name = predictor.predict(N, P, K, temperature, humidity, ph, rainfall)
        print(f"Predicted class: {class_name}")
        class_index = predictor.label_encoder.transform([class_name])[0]

        # Preprocess input sample
        X_processed = predictor.preprocess_data(X, fit=False)

        # Get SHAP values
        shap_values = explainer.shap_values(X_processed)
        sample_idx = 0  # Assuming only one sample
        sample_shap_values = shap_values[:, :, class_index][sample_idx]  # SHAP values for the specific class
        sample_data = X[sample_idx]  # Original input data

        # Feature names
        feature_names = predictor.numerical_features

        # Validate shapes
        if len(sample_shap_values) != len(feature_names):
            raise ValueError("Mismatch between SHAP values and feature names.")

        # Create SHAP explanation object
        explanation = shap.Explanation(
            values=sample_shap_values,
            base_values=explainer.expected_value[class_index],
            data=sample_data,
            feature_names=feature_names,
        )

        # Plot SHAP diagram
        plt.figure()
        if plot_type == "waterfall":
            shap.plots.waterfall(explanation, show=False)
        elif plot_type == "force":
            shap.plots.force(explanation,  show=False)
        plt.subplots_adjust(left=0.3)
        # Save plot and return encoded image
        plot_filename = f"shap_plot_{plot_type}_class_{class_name}.png"
        plt.savefig(plot_filename)
        plt.close()
        encode_img = SHAPUtility.encode_img(plot_filename)
        return encode_img



    @staticmethod
    def get_shap_global_explanation(predictor, explainer, label, num_samples=100):
        """
        Generate a global SHAP summary plot.
        """
        # Prepare test data without the label
        X_test = predictor.test_data.drop(['label'], axis=1)
        X_test_processed = predictor.preprocess_data(X_test, fit=False)[:num_samples]
        
        class_index = predictor.label_encoder.transform([label])[0]
        # Get SHAP values
        shap_values = explainer.shap_values(X_test_processed)
        shap_values = shap_values[:,:,class_index]  # SHAP values for the specific class
        # Check if shap_values is a list (for models with multiple outputs)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Use the first output (adapt if necessary)
        
        # Ensure shap_values and X_test_processed have consistent shapes
        if shap_values.shape[1] != X_test_processed.shape[1]:
            raise ValueError(
                f"Mismatch between SHAP values ({shap_values.shape[1]} features) "
                f"and processed input data ({X_test_processed.shape[1]} features)."
            )
        
        # Align feature names with the processed data
        feature_names = predictor.numerical_features
        if len(feature_names) != X_test_processed.shape[1]:
            raise ValueError(
                f"Mismatch between feature names ({len(feature_names)}) "
                f"and processed input data ({X_test_processed.shape[1]} features)."
            )
        
        # Generate the SHAP summary plot
        plt.figure()
        shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, show=False)
        
        # Save the plot
        return SHAPUtility._save_plot(f"shap_global_summary_{label}")

    @staticmethod
    def save_explainer(explainer, file_path):
        """
        Save the SHAP explainer to a file.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(explainer, f)
        print(f"SHAP explainer saved to {file_path}.")

    @staticmethod
    def load_explainer(file_path):
        """
        Load the SHAP explainer from a file.
        """
        with open(file_path, 'rb') as f:
            explainer = pickle.load(f)
        print(f"SHAP explainer loaded from {file_path}.")
        return explainer

    @staticmethod
    def encode_img(file_path):
        """
        Encode an image file as Base64.
        """
        with open(file_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def _save_plot(file_prefix):
        """
        Save the current matplotlib plot to a file and return the Base64 encoding.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # file_path = os.path.join("images", f"{file_prefix}_{timestamp}.png")
        # file_path = os.path.join("images", f"{file_prefix}.png")
        file_path = f"{file_prefix}.png"
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()

        with open(file_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

if __name__ == "__main__":
    

    
    # Input to explain
    # X = [[2, 5, 12, 5.2, 2.3, 1.1, 12.5]]
    input_data = {
    
    "N": 50,
    "P": 30,
    "K": 20,
    "ph": 6.5,
    "temperature": 25.0,
    "humidity": 60.5,
    "rainfall": 100.0,
    "plot_type": "waterfall"
}
    SHAPUtility.get_shap_global_explanation(CropPredictor(), SHAPUtility.initialize_shap_explainer(CropPredictor()), num_samples=150)
    # # Class to explain
    # class_name = "beans"
    # class_name = None
    
    # # Generate SHAP diagram
    # shap_diagram_enc = SHAPUtility.generate_shap_diagram(input_data)
    # print(shap_diagram_enc)


    