import shap
import matplotlib.pyplot as plt
import numpy as np
import base64
import datetime



def get_shap_values(
    X: list, predictor
):
    """
    Get SHAP values for the given input data using the provided model.
    """


    # Preprocess the input data using the model's preprocessor
    X_processed = predictor.preprocess_data(X, fit=False)

    # Initialize SHAP masker and explainer
    masker = shap.maskers.Independent(X_processed)
    feature_names = list(predictor.train_data.columns[:-1])  # Exclude 'disease'
    explainer = shap.Explainer(predictor.predict, masker, feature_names=feature_names)
    shap_values = explainer(X_processed)
    return shap_values

def get_shap_diagram(
    X: list, predictor,plot_type="waterfall"
):
    """
    Get a diagram for the SHAP values.
    """

    shap_values = get_shap_values(
      X, predictor
    )

    # Plot SHAP values based on the selected plot type
    if plot_type == "waterfall":
        shap.plots.waterfall(shap_values[0], show=False)
    elif plot_type == "force":
        shap.plots.force(shap_values[0], show=False)
    else:
        raise ValueError("Invalid plot type. Please choose between 'waterfall' and 'force'.")

    plt_path = "shap_plot_" + str(datetime.datetime.now()) + ".png"
    plt.savefig(plt_path)
    plt.close()  # Close plot to avoid memory issues


    with open(plt_path, 'rb') as image_file:
        base64_encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_encoded_image
