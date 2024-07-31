import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
import os
import time
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from concurrent.futures import ThreadPoolExecutor

REPEAT_RUNS = 1

class HeartDiseasePredictor:
    def __init__(self):
        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.nn_model = None
        self.preprocessor = None
        self.weights = {
            'age': 1.0, 'sex': 1.0, 'cp': 1.0, 'trestbps': 1.0, 'chol': 1.0,
            'fbs': 1.0, 'restecg': 1.0, 'thalach': 1.0, 'exang': 1.0,
            'oldpeak': 1.0, 'slope': 1.0, 'ca': 1.0, 'thal': 1.0
        }

    def load_data_from_directory(self, directory="heart+disease"):
        files = [
            "processed.cleveland.data",
            "processed.hungarian.data",
            "processed.switzerland.data",
            "processed.va.data"
        ]

        def load_and_process_file(file):
            file_path = os.path.join(directory, file)
            if not os.path.exists(file_path):
                print(f"[ERR] File not found: {file_path}")
                return None
            try:
                data = pd.read_csv(file_path, header=None, na_values='?', sep=',')
                columns = [*self.weights]
                columns.append('disease')
                data.columns = columns
                
                for col in data.columns:
                    data[col] = data[col].astype(float)
                
                data['disease'] = data['disease'].apply(lambda x: 1 if x > 0 else 0) # ToDo: Binary Prediction or numerical?
                data.dropna(how='any', inplace=True) # ToDo: Drop '?' or replace with valid values?
                return data
            except pd.errors.ParserError as e:
                print(f"[ERR] Error parsing {file_path}: {e}")
                return None

        with ThreadPoolExecutor() as executor:
            all_data = list(executor.map(load_and_process_file, files))
        
        all_data = [data for data in all_data if data is not None]

        if not all_data:
            print("[ERR] No valid data loaded.")
            return

        combined_data = pd.concat(all_data, ignore_index=True)
        self.train_data, self.test_data = train_test_split(combined_data, test_size=0.2, random_state=42)
        self.train_nn_model()

    def build_nn_model(self, input_shape, layers_config):
        model = Sequential()
        model.add(Input(shape=input_shape))
        for i, layer in enumerate(layers_config):
            units, activation, dropout = layer['units'], layer['activation'], layer.get('dropout', 0)
            model.add(Bidirectional(LSTM(units, activation=activation, return_sequences=(i < len(layers_config) - 1))))
            if dropout > 0:
                model.add(Dropout(dropout))
        
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def preprocess_data(self, data, fit=True):
        categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
        numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

        if fit:
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OneHotEncoder(), categorical_features)
                ])
            processed_data = self.preprocessor.fit_transform(data)
        else:
            processed_data = self.preprocessor.transform(data)

        return processed_data

    def weighted_loss(self, y_true, y_pred):
        weights = tf.constant([self.weights[feature] for feature in self.train_data.columns if feature != 'disease'], dtype=tf.float32)
        loss = tf.reduce_mean(tf.multiply(weights, tf.square(y_true - y_pred)))
        return loss

    def train_nn_model(self):
        X = self.train_data.drop(['disease'], axis=1)
        y = self.train_data['disease']

        X_processed = self.preprocess_data(X)
        X_processed = np.expand_dims(X_processed, axis=1)

        layers_config = [
            {'units': 64, 'activation': 'relu', 'dropout': 0.2},
            {'units': 32, 'activation': 'relu'}
        ]

        self.nn_model = self.build_nn_model((X_processed.shape[1], X_processed.shape[2]), layers_config)
        self.nn_model.compile(loss=self.weighted_loss, optimizer='adam', metrics=['accuracy'])
        self.nn_model.fit(X_processed, y, epochs=32, batch_size=64, verbose=1, validation_split=0.2)
        self.run_nn_simulation()

    def run_nn_simulation(self):
        if self.test_data.empty:
            print("[ERR] Test dataset is empty. Load the data first.")
            return
        if self.nn_model is None:
            print("[ERR] Neural network is not trained. Train the neural network first.")
            return

        X_test = self.test_data.drop(['disease'], axis=1)
        y_test = self.test_data['disease']

        X_test_processed = self.preprocess_data(X_test, fit=False)
        X_test_processed = np.expand_dims(X_test_processed, axis=1)

        y_pred_prob = self.nn_model.predict(X_test_processed)
        y_pred = (y_pred_prob > 0.5).astype(int)

        y_test_df = self.test_data.copy()
        y_test_df['predicted'] = y_pred
        y_test_df['correct'] = y_test_df['disease'] == y_test_df['predicted']

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        if REPEAT_RUNS == 1:
            print("Result:")
            print(y_test_df[['disease', 'predicted', 'correct']])
            print(classification_report(y_test, y_pred))

        return accuracy, f1
    
    def predict(self, X):
        X_processed = self.preprocess_data(X, fit=False)
        X_processed = np.expand_dims(X_processed, axis=1)

        y_prob = self.nn_model.predict(X_processed)
        y = (y_prob > 0.5).astype(int)

        return y
    
    def set_weight(self, feature, weight):
        if feature in self.weights:
            self.weights[feature] = weight
            print(f"Weight for feature '{feature}' set to {weight}.")
        else:
            print(f"[ERR] Feature '{feature}' not found.")

    def calculate_correlation(self, x1, x2):
        if x1 in self.train_data.columns and x2 in self.train_data.columns:
            correlation = self.train_data[x1].corr(self.train_data[x2])
            print(f"Correlation between {x1} and {x2}: {correlation}")
            return correlation
        else:
            print(f"[ERR] One or both features '{x1}' and '{x2}' not found.")
            return None
    
def main(repeat_runs=0):
    predictor = HeartDiseasePredictor()

    # Method Unit tests

    times = []
    accuracies = []
    f1_scores = []

    if repeat_runs == 1:
        # Run a normal simulation
        predictor.load_data_from_directory()
        accuracy, f1 = predictor.run_nn_simulation()
        
        # Select first dataset and predict
        # first_test_sample = predictor.test_data.iloc[[0]].drop(['disease'], axis=1)
        # y = predictor.predict(first_test_sample)
        # print(f"Prediction for the first test sample: {y[0]} - {predictor.test_data.iloc[[0]]['disease']}")

        # Calculate correlation between two random features
        x1, x2 = 'age', 'disease'
        predictor.calculate_correlation(x1, x2)

        # Set weight for a specific feature and retrain the model
        # predictor.set_weight('age', 4)
        # predictor.train_nn_model()
        # accuracy, f1 = predictor.run_nn_simulation()
    else:
        for _ in range(repeat_runs):
            start_time = time.time()
            predictor.load_data_from_directory()
            accuracy, f1 = predictor.run_nn_simulation()
            end_time = time.time()

            times.append(end_time - start_time)
            accuracies.append(accuracy)
            f1_scores.append(f1)

        avg_time = np.mean(times)
        avg_accuracy = np.mean(accuracies)
        avg_f1 = np.mean(f1_scores)
        print(f"Average Execution Time: {avg_time} seconds")
        print(f"Average Accuracy: {avg_accuracy}")
        print(f"Average F1 Score: {avg_f1}")

if __name__ == "__main__":
    tf.config.optimizer.set_jit(False)
    
    main(repeat_runs=REPEAT_RUNS)
