import shap
from model_interface import PredictionModel
import matplotlib.pyplot as plt
from icecream import ic
import numpy as np
from io import BytesIO
import base64
import datetime
def load_model(path=None):
    """
    Load the model for the SHAP values.
    """
    if path is None:
        model = PredictionModel()
        model.train()
    else:
        model = PredictionModel.load_model(path)
    
    return model

def get_shap_values(x1,x2,x3, background_data=None):
    """
    Get SHAP values for the given  input data.
    """
    model = load_model()
    X = np.array([x1, x2, x3])
    if background_data is None:
        # Use the mean of X as background if not provided
        background_data = model.X[:100]  # Use first 100 samples as background
    
    masker = shap.maskers.Independent(background_data)
    feature_names = ["Feature 1", "Feature 2", "Feature 3"]
    explainer = shap.Explainer(model, masker,feature_names=feature_names)
    shap_values = explainer(X)
    print(type(shap_values))
    return shap_values

def get_shap_diagram(x1,x2,x3, plot_type="waterfall", encode=True):
    """
    Get a diagram for the SHAP values.
    """
    X = np.array([x1, x2, x3])
    shap_values = get_shap_values(X)

    if plot_type == "waterfall":
        shap.plots.waterfall(shap_values[0], show=False)
    elif plot_type == "force":
        shap.plots.force(shap_values[0], show=False)
    else:
        raise ValueError("Invalid plot type. Please choose between 'waterfall' and 'force'.")
    plt_path = "shap_plot_" + str(datetime.datetime.now()) + ".png"
    plt.savefig(plt_path)
    if encode:
        with open(plt_path, 'rb') as image_file:
            base64_encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            return base64_encoded_image
    else:
        return plt_path



