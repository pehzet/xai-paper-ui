import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

FORCE_TRAIN = 0


class CropPredictor:
    """
    A class for predicting crop using a neural network model.
    """

    def __init__(self):
        """
        Initializes the CropPredictor instance.

        Sets up data structures, initializes feature loads data,
        trains the initial model, and runs the initial simulation.
        """
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.num_classes = None

        self.numerical_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

        self.model_path = os.path.join('prediction_model', 'model', 'nn_model.keras')
        

        self.load_data()
        if not os.path.exists(self.model_path) or FORCE_TRAIN:
            self.train_model()
            self.save_model()
        else:
            self.model = tf.keras.models.load_model(self.model_path)
            self.initialize_scaler()

        # self.run_simulation()

    def load_data(self, filepath=os.path.join('data', 'Crop_recommendation.csv')):
        """
        Loads and preprocesses the dataset from the given CSV file.

        Parameters:
        - filepath (str): The path to the CSV file containing the dataset.

        Returns:
        - None
        """
        try:
            filepath = os.path.join(os.path.dirname(__file__), filepath)
            data = pd.read_csv(filepath)

            data[self.numerical_features] = data[self.numerical_features].fillna(data[self.numerical_features].mean())

            self.label_encoder = LabelEncoder()
            data['label'] = self.label_encoder.fit_transform(data['label'])
            self.num_classes = len(self.label_encoder.classes_)

            self.train_data, self.test_data = train_test_split(
                data, test_size=0.5, random_state=42, stratify=data['label']
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
        - X_scaled (numpy.ndarray): The preprocessed data.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values  # Convert to NumPy array for consistent processing
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled

    def build_model(self, input_dim, num_classes):
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

    def train_model(self):
        """
        Trains the neural network model using the training data.

        Returns:
        - None
        """

        X_processed = self.preprocess_data(self.X_train, fit=True)
        input_dim = X_processed.shape[1]
        num_classes = self.num_classes

        model = self.build_model(input_dim, num_classes)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        model.fit(
            X_processed, self.y_train, epochs=10, batch_size=64, verbose=0,
            validation_split=0.2, callbacks=[early_stopping]
        )

        self.model = model

    def save_model(self):
        """
        Saves the trained model to a file.
        """
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)
            print(f"[INFO] Model saved to {self.model_path}")
        except Exception as e:
            print(f"[ERR] Error saving model: {e}")

    def initialize_scaler(self):
        """
        Initializes the scaler using the training data.
        """
        self.scaler = StandardScaler()
        self.scaler.fit(self.X_train)  # Fit scaler on training data
        print("[INFO] Scaler initialized.")

    def run_simulation(self):
        """
        Evaluates the current neural network model on the test dataset and prints the classification report.

        Returns:
        - None
        """
        if self.model is None:
            print("[ERR] The neural network is not trained. Please train it first.")
            return

        X_test_processed = self.preprocess_data(self.X_test, fit=False)

        y_pred_prob = self.model.predict(X_test_processed, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)

        return classification_report(self.y_test, y_pred, target_names=self.label_encoder.classes_)

    def predict(self, N, P, K, temperature, humidity, ph, rainfall):
        """
        Predicts the class for the given input data.

        Parameters:
        - Each feature from dataset

        Returns:
        - y (numpy.ndarray): The predicted class labels.
        """
        X = np.array([N, P, K, temperature, humidity, ph, rainfall]).reshape(1, -1)

        X_processed = self.preprocess_data(X, fit=False)

        y_prob = self.model.predict(X_processed, verbose=0)
        y = np.argmax(y_prob, axis=1)
        return self.label_encoder.inverse_transform(y)

    def sum_feature(self, feature, by_class=False):
        """
        Calculates the sum of a feature across the entire dataset or grouped by class.

        Parameters:
        - feature (str): Name of the feature.
        - by_class (bool): If True, calculates the sum grouped by class.

        Returns:
        - sum (float or dict): Total sum or sum per class.
        """
        if by_class:
            return self.train_data.groupby('label')[feature].sum().to_dict()
        return self.train_data[feature].sum()
    
    def mean_feature(self, feature, by_class=False):
        """
        Calculates the mean of a feature across the entire dataset or grouped by class.

        Parameters:
        - feature (str): Name of the feature.
        - by_class (bool): If True, calculates the mean grouped by class.

        Returns:
        - mean (float or dict): Overall mean or mean per class.
        """
        if by_class:
            return self.train_data.groupby('label')[feature].mean().to_dict()
        return self.train_data[feature].mean()
    
    def quantile_feature(self, feature, quantiles=[0.25, 0.5, 0.75]):
        """
        Calculates specified quantiles of a feature.

        Parameters:
        - feature (str): Name of the feature.
        - quantiles (list): List of desired quantiles.

        Returns:
        - result (dict): Dictionary of quantiles and their values.
        """
        return self.train_data[feature].quantile(quantiles).to_dict()
    
    def variance_feature(self, feature):
        """
        Calculates the variance of a feature.

        Parameters:
        - feature (str): Name of the feature.

        Returns:
        - variance (float): Variance of the feature.
        """
        return self.train_data[feature].var()

    def std_feature(self, feature):
        """
        Calculates the standard deviation of a feature.

        Parameters:
        - feature (str): Name of the feature.

        Returns:
        - std (float): Standard deviation of the feature.
        """
        return self.train_data[feature].std()
    
    def min_feature(self, feature):
        """
        Returns the minimum value of a feature.

        Parameters:
        - feature (str): Name of the feature.

        Returns:
        - min_value (float): Minimum value of the feature.
        """
        return self.train_data[feature].min()

    def max_feature(self, feature):
        """
        Returns the maximum value of a feature.

        Parameters:
        - feature (str): Name of the feature.

        Returns:
        - max_value (float): Maximum value of the feature.
        """
        return self.train_data[feature].max()
    
    def correlation(self, feature1, feature2):
        """
        Calculates the correlation coefficient between two features.

        Parameters:
        - feature1 (str): Name of the first feature.
        - feature2 (str): Name of the second feature.

        Returns:
        - correlation (float): Correlation coefficient between the two features.
        """
        return self.train_data[[feature1, feature2]].corr().iloc[0, 1]
    
    def class_distribution(self):
        """
        Returns the frequency of each class in the dataset.

        Returns:
        - distribution (dict): Dictionary with class labels and their frequencies.
        """
        return self.train_data['label'].value_counts().to_dict()
    
    def feature_values_for_class(self, feature, class_label):
        """
        Returns the values of a feature for a specific class.

        Parameters:
        - feature (str): Name of the feature.
        - class_label (str or int): Name or index of the class.

        Returns:
        - values (list): List of feature values for the given class.
        """
        class_index = self.label_encoder.transform([class_label])[0]
        return self.train_data[self.train_data['label'] == class_index][feature].tolist()
    
    def feature_distribution(self, feature, bins=10):
        """
        Returns the histogram distribution of a feature.

        Parameters:
        - feature (str): Name of the feature.
        - bins (int): Number of bins for the histogram.

        Returns:
        - hist (dict): Dictionary with bin edges and counts.
        """
        counts, bin_edges = np.histogram(self.train_data[feature], bins=bins)
        return dict(zip(bin_edges, counts))

# def main():
#     predictor = CropPredictor()
#     y = predictor.predict(2, 5, 12, 5.2, 2.3, 1.1, 12.5)
#     print(y)

# if __name__ == "__main__":
#     main()