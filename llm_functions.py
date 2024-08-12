from neuronal_network import HeartDiseasePredictor
import pandas as pd
import numpy as np
from shap_utils import get_shap_diagram as _get_shap_diagram, get_shap_values as _get_shap_values
import matplotlib.pyplot as plt
predictor = HeartDiseasePredictor()
X_default = { # random values
    'age': 50,
    'sex': 1,
    'cp': 0,
    'trestbps': 120,
    'chol': 200,
    'fbs': 0,
    'restecg': 0,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 1.0,
    'slope': 1,
    'ca': 0,
    'thal': 2
}

def load_model():

    # model = predictor.load_last_model() # TODO: implement
    model = predictor.nn_model
    return model

def predict(age=None, sex=None, cp=None, trestbps=None, chol=None, fbs=None, restecg=None, thalach=None, exang=None, oldpeak=None, slope=None, ca=None, thal=None):
    # X = [
    #     age if age is not None else X_default['age'],
    #     sex if sex is not None else X_default['sex'],
    #     cp if cp is not None else X_default['cp'],
    #     trestbps if trestbps is not None else X_default['trestbps'],
    #     chol if chol is not None else X_default['chol'],
    #     fbs if fbs is not None else X_default['fbs'],
    #     restecg if restecg is not None else X_default['restecg'],
    #     thalach if thalach is not None else X_default['thalach'],
    #     exang if exang is not None else X_default['exang'],
    #     oldpeak if oldpeak is not None else X_default['oldpeak'],
    #     slope if slope is not None else X_default['slope'],
    #     ca if ca is not None else X_default['ca'],
    #     thal if thal is not None else X_default['thal']
    # ]
    X = {
        'age': age if age is not None else X_default['age'],
        'sex': sex if sex is not None else X_default['sex'],
        'cp': cp if cp is not None else X_default['cp'],
        'trestbps': trestbps if trestbps is not None else X_default['trestbps'],
        'chol': chol if chol is not None else X_default['chol'],
        'fbs': fbs if fbs is not None else X_default['fbs'],
        'restecg': restecg if restecg is not None else X_default['restecg'],
        'thalach': thalach if thalach is not None else X_default['thalach'],
        'exang': exang if exang is not None else X_default['exang'],
        'oldpeak': oldpeak if oldpeak is not None else X_default['oldpeak'],
        'slope': slope if slope is not None else X_default['slope'],
        'ca': ca if ca is not None else X_default['ca'],
        'thal': thal if thal is not None else X_default['thal']
    }
    X = pd.DataFrame(X, columns=X_default.keys())
    Y = predictor.predict(X)
    # maybe format the output for LLM
    return Y

def train_nn_model(layers_config):
    # layers_config = [
    #             {'units': 64, 'activation': 'relu', 'dropout': 0.2},
    #             {'units': 32, 'activation': 'relu'}
    # ]        
    # TODO: fallback wrong layers config
    new_model = predictor.train_nn_model(layers_config) # todo: return model
    predictor.models.append(new_model) # todo: models array and access model in code, default model = models[0]

def get_confidence():
    # TODO: implement confidence of last (?) prediction
    pass

def adjust_dataset_and_train_nn(column_weights):
    # column_weights = {'age': 0.5}
    pass
def get_correlation(x1,x2):
    # x1 = 'age'
    # x2 = 'sex'
    pass


# SHAP FUNCTIONS

def get_shap_values(
    age=None, sex=None, cp=None, trestbps=None, chol=None, fbs=None,
    restecg=None, thalach=None, exang=None, oldpeak=None, slope=None, ca=None, thal=None,
    model=None
):
    """
    Get SHAP values for the given input data using the provided model.
    """
    X = {
        'age': age if age is not None else X_default['age'],
        'sex': sex if sex is not None else X_default['sex'],
        'cp': cp if cp is not None else X_default['cp'],
        'trestbps': trestbps if trestbps is not None else X_default['trestbps'],
        'chol': chol if chol is not None else X_default['chol'],
        'fbs': fbs if fbs is not None else X_default['fbs'],
        'restecg': restecg if restecg is not None else X_default['restecg'],
        'thalach': thalach if thalach is not None else X_default['thalach'],
        'exang': exang if exang is not None else X_default['exang'],
        'oldpeak': oldpeak if oldpeak is not None else X_default['oldpeak'],
        'slope': slope if slope is not None else X_default['slope'],
        'ca': ca if ca is not None else X_default['ca'],
        'thal': thal if thal is not None else X_default['thal']
    }
    model = load_model()
    
    # Assemble input features into a single sample
    X = pd.DataFrame([X], columns=X_default.keys())
    shap_values = _get_shap_values(X, predictor)

    return shap_values

def get_shap_diagram(
    age=None, sex=None, cp=None, trestbps=None, chol=None, fbs=None,
    restecg=None, thalach=None, exang=None, oldpeak=None, slope=None, ca=None, thal=None,
    model=None, plot_type="waterfall"
):
    """
    Get a diagram for the SHAP values.
    """
    X = {
        'age': age if age is not None else X_default['age'],
        'sex': sex if sex is not None else X_default['sex'],
        'cp': cp if cp is not None else X_default['cp'],
        'trestbps': trestbps if trestbps is not None else X_default['trestbps'],
        'chol': chol if chol is not None else X_default['chol'],
        'fbs': fbs if fbs is not None else X_default['fbs'],
        'restecg': restecg if restecg is not None else X_default['restecg'],
        'thalach': thalach if thalach is not None else X_default['thalach'],
        'exang': exang if exang is not None else X_default['exang'],
        'oldpeak': oldpeak if oldpeak is not None else X_default['oldpeak'],
        'slope': slope if slope is not None else X_default['slope'],
        'ca': ca if ca is not None else X_default['ca'],
        'thal': thal if thal is not None else X_default['thal']
    }
    X = pd.DataFrame(X, columns=X_default.keys())
    if model is None:
        model = load_model()

    shap_diagram_encoded = _get_shap_diagram(X, predictor, plot_type=plot_type)
    return shap_diagram_encoded