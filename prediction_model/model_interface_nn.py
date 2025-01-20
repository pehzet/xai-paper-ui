# model_interface.py
import json
from .neural_network import CropPredictor
import numpy as np
from icecream import ic
# Initialize the Crop Predictor model
predictor = CropPredictor()

def _numpy_to_native(data):
    ic(data)
    """
    Convert NumPy data types to native Python data types.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: _numpy_to_native(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_numpy_to_native(v) for v in data]
    elif isinstance(data,  np.int64):
        return int(data)
    elif isinstance(data,  np.float64):
        return float(data)
    else:
        return data

def predict(data):
    """
    Predicts the crop based on the soil and environmental characteristics.
    """
    try:
        # Handle input data (string or dictionary)
        if isinstance(data, str):
            data = json.loads(data)
        elif not isinstance(data, dict):
            raise ValueError(f"Invalid data type: {type(data)}")

        # Extract required parameters
        N = data["N"]
        P = data["P"]
        K = data["K"]
        temperature = data["temperature"]
        humidity = data["humidity"]
        ph = data["ph"]
        rainfall = data["rainfall"]

        # Make prediction
        prediction = predictor.predict(N, P, K, temperature, humidity, ph, rainfall)
        # Return the first predicted label
        return _numpy_to_native(prediction[0])

    except Exception as e:
        return {"error": str(e)}



def sum_feature(data):
    """
    Calculates the sum of a feature across the entire dataset or grouped by class.
    """
    try:
        if isinstance(data, str):
            data = json.loads(data)
        elif not isinstance(data, dict):
            raise ValueError(f"Invalid data type: {type(data)}")

        feature = data["feature"]
        by_class = data.get("by_class", False)

        result = predictor.sum_feature(feature, by_class=by_class)
        return _numpy_to_native(result)

    except Exception as e:
        return {"error": str(e)}

def mean_feature(data):
    """
    Calculates the mean of a feature across the entire dataset or grouped by class.
    """
    try:
        if isinstance(data, str):
            data = json.loads(data)
        elif not isinstance(data, dict):
            raise ValueError(f"Invalid data type: {type(data)}")

        feature = data["feature"]
        by_class = data.get("by_class", False)

        result = predictor.mean_feature(feature, by_class=by_class)
        return _numpy_to_native(result)

    except Exception as e:
        return {"error": str(e)}

def quantile_feature(data):
    """
    Calculates specified quantiles of a feature.
    """
    try:
        if isinstance(data, str):
            data = json.loads(data)
        elif not isinstance(data, dict):
            raise ValueError(f"Invalid data type: {type(data)}")

        feature = data["feature"]
        quantiles = data.get("quantiles", [0.25, 0.5, 0.75])

        result = predictor.quantile_feature(feature, quantiles=quantiles)
        return _numpy_to_native(result)

    except Exception as e:
        return {"error": str(e)}

def variance_feature(data):
    """
    Calculates the variance of a feature.
    """
    try:
        if isinstance(data, str):
            data = json.loads(data)
        elif not isinstance(data, dict):
            raise ValueError(f"Invalid data type: {type(data)}")

        feature = data["feature"]
        result = predictor.variance_feature(feature)
        return _numpy_to_native(result)

    except Exception as e:
        return {"error": str(e)}

def std_feature(data):
    """
    Calculates the standard deviation of a feature.
    """
    try:
        if isinstance(data, str):
            data = json.loads(data)
        elif not isinstance(data, dict):
            raise ValueError(f"Invalid data type: {type(data)}")

        feature = data["feature"]
        result = predictor.std_feature(feature)
        return _numpy_to_native(result)

    except Exception as e:
        return {"error": str(e)}

def min_feature(data):
    """
    Returns the minimum value of a feature.
    """
    try:
        if isinstance(data, str):
            data = json.loads(data)
        elif not isinstance(data, dict):
            raise ValueError(f"Invalid data type: {type(data)}")

        feature = data["feature"]
        result = predictor.min_feature(feature)
        return _numpy_to_native(result)

    except Exception as e:
        return {"error": str(e)}

def max_feature(data):
    """
    Returns the maximum value of a feature.
    """
    try:
        if isinstance(data, str):
            data = json.loads(data)
        elif not isinstance(data, dict):
            raise ValueError(f"Invalid data type: {type(data)}")

        feature = data["feature"]
        result = predictor.max_feature(feature)
        return _numpy_to_native(result)

    except Exception as e:
        return {"error": str(e)}

def correlation(data):
    """
    Calculates the correlation coefficient between two features.
    """
    try:
        if isinstance(data, str):
            data = json.loads(data)
        elif not isinstance(data, dict):
            raise ValueError(f"Invalid data type: {type(data)}")

        feature1 = data["feature1"]
        feature2 = data["feature2"]
        result = predictor.correlation(feature1, feature2)
        return _numpy_to_native(result)

    except Exception as e:
        return {"error": str(e)}

def class_distribution(data):
    """
    Returns the frequency of each class in the dataset.
    """
    try:
        if isinstance(data, str):
            data = json.loads(data)
        elif not isinstance(data, dict):
            raise ValueError(f"Invalid data type: {type(data)}")

        result = predictor.class_distribution()
        return _numpy_to_native(result)

    except Exception as e:
        return {"error": str(e)}

def feature_values_for_class(data):
    """
    Returns the values of a feature for a specific class.
    """
    try:
        if isinstance(data, str):
            data = json.loads(data)
        elif not isinstance(data, dict):
            raise ValueError(f"Invalid data type: {type(data)}")

        feature = data["feature"]
        class_label = data["class_label"]
        result = predictor.feature_values_for_class(feature, class_label)
        return _numpy_to_native(result)

    except Exception as e:
        return {"error": str(e)}

def feature_distribution(data):
    """
    Returns the histogram distribution of a feature.
    """
    try:
        if isinstance(data, str):
            data = json.loads(data)
        elif not isinstance(data, dict):
            raise ValueError(f"Invalid data type: {type(data)}")

        feature = data["feature"]
        bins = data.get("bins", 10)
        result = predictor.feature_distribution(feature, bins=bins)
        return _numpy_to_native(result)

    except Exception as e:
        return {"error": str(e)}

def run_simulation(data):
    """
    Runs a simulation and returns a classification report.
    """
    try:
        if isinstance(data, str):
            data = json.loads(data)
        elif not isinstance(data, dict):
            raise ValueError(f"Invalid data type: {type(data)}")

        report = predictor.run_simulation()
        return _numpy_to_native(report)

    except Exception as e:
        return {"error": str(e)}
