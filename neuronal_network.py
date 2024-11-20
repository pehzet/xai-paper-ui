import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

class CropPredictor:
    """
    A class for predicting crop using a neural network model.
    """

    def __init__(self):
        """
        Initializes the CropPredictor instance.

        Sets up data structures, initializes feature weights, loads data,
        trains the initial model, and runs the initial simulation.
        """
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.nn_model = None
        self.models = []
        self.preprocessor = None
        self.label_encoder = None
        self.num_classes = None

        self.numerical_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

        self.weights = {feature: 1.0 for feature in self.numerical_features}

        self.load_data()
        self.train_nn_model()
        self.run_nn_simulation()

    def load_data(self, filepath=r'xai-paper-ui\data\Crop_recommendation.csv'):
        """
        Loads and preprocesses the dataset from the given CSV file.

        Parameters:
        - filepath (str): The path to the CSV file containing the dataset.

        Returns:
        - None
        """
        try:
            data = pd.read_csv(filepath)

            data[self.numerical_features] = data[self.numerical_features].fillna(data[self.numerical_features].mean())

            self.label_encoder = LabelEncoder()
            data['label'] = self.label_encoder.fit_transform(data['label'])
            self.num_classes = len(self.label_encoder.classes_)

            self.train_data, self.test_data = train_test_split(
                data, test_size=0.6, random_state=42, stratify=data['label']
            )

            self.X_train = self.train_data[self.numerical_features].values
            self.y_train = self.train_data['label'].values
            self.X_test = self.test_data[self.numerical_features].values
            self.y_test = self.test_data['label'].values

        except Exception as e:
            print(f"[ERR] Error loading data: {e}")

    def preprocess_data(self, X, fit=True):
        """
        Preprocesses the input data by applying scaling and feature weighting.

        Parameters:
        - X (numpy.ndarray): The input data to preprocess.
        - fit (bool, optional): Whether to fit the scaler to the data. Default is True.

        Returns:
        - processed_data (numpy.ndarray): The preprocessed data.
        """
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        weight_vector = np.array([self.weights.get(name, 1.0) for name in self.numerical_features])
        processed_data = X_scaled * weight_vector

        return processed_data

    def build_nn_model(self, input_dim, num_classes):
        """
        Builds and compiles a neural network model.

        Parameters:
        - input_dim (int): The number of input features.
        - num_classes (int): The number of output classes.

        Returns:
        - model (tensorflow.keras.models.Sequential): The compiled neural network model.
        """
        model = Sequential(
            layers=[
                tf.keras.layers.InputLayer(shape=(input_dim,)),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(num_classes, activation='softmax')
            ]
        )

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train_nn_model(self):
        """
        Trains the neural network model using the training data.

        Returns:
        - None
        """
        X_processed = self.preprocess_data(self.X_train, fit=True)
        input_dim = X_processed.shape[1]
        num_classes = self.num_classes

        model = self.build_nn_model(input_dim, num_classes)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        model.fit(
            X_processed, self.y_train, epochs=8, batch_size=64, verbose=0,
            validation_split=0.2, callbacks=[early_stopping]
        )

        self.models.append(model)
        self.nn_model = model

    def run_nn_simulation(self):
        """
        Evaluates the current neural network model on the test dataset and prints the classification report.

        Returns:
        - None
        """
        if self.nn_model is None:
            print("[ERR] The neural network is not trained. Please train it first.")
            return

        X_test_processed = self.preprocess_data(self.X_test, fit=False)

        y_pred_prob = self.nn_model.predict(X_test_processed, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)

        print("Simulation Result:")
        print(classification_report(self.y_test, y_pred, target_names=self.label_encoder.classes_))

    def predict(self, X, model_index=None):
        """
        Predicts the class for the given input data.

        Parameters:
        - X (numpy.ndarray or pandas.DataFrame): The input data for prediction.
        - model_index (int, optional): Index of the model to use from the models list. If None, uses the latest model.

        Returns:
        - y (numpy.ndarray): The predicted class labels.
        """
        if isinstance(X, pd.DataFrame):
            X = X[self.numerical_features].values
        else:
            X = np.array(X)

        X_processed = self.preprocess_data(X, fit=False)
        if X_processed.ndim == 1:
            X_processed = X_processed.reshape(1, -1)

        if model_index is not None and 0 <= model_index < len(self.models):
            model = self.models[model_index]
        else:
            model = self.nn_model

        y_prob = model.predict(X_processed, verbose=0)
        y = np.argmax(y_prob, axis=1)
        return self.label_encoder.inverse_transform(y)

    def set_weight(self, feature, weight):
        """
        Sets the weight for a specific feature and retrains the neural network model.

        Parameters:
        - feature (str): The name of the feature to adjust the weight.
        - weight (float): The new weight value for the feature.

        Returns:
        - None
        """
        if feature in self.weights:
            self.weights[feature] = weight
            print(f"Weight for feature '{feature}' set to {weight}.")
            self.train_nn_model()
            self.run_nn_simulation()
        else:
            print(f"[ERR] Feature '{feature}' not found.")

def main():
    start_time = time.time()
    predictor = CropPredictor()

    # Example usage:
    # predictor.set_weight('N', 1.5)
    # sample_input = predictor.test_data.iloc[[0]]
    # prediction = predictor.predict(sample_input)
    # print(f"Prediction: {prediction[0]}")

    print(f"Execution time: {time.time() - start_time:.4f} seconds")

if __name__ == "__main__":
    main()
