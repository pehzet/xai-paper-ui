import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Unterdrückt INFO-Meldungen von TensorFlow

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from concurrent.futures import ThreadPoolExecutor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.base import TransformerMixin, BaseEstimator

class FeatureWeigher(TransformerMixin, BaseEstimator):
    def __init__(self, weight):
        self.weight = weight

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X * self.weight

class ColumnSelector(TransformerMixin, BaseEstimator):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]

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

        self.numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        self.categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'ca']

        self.load_data_from_directory()
        self.train_nn_model()

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
                columns = [
                    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'disease'
                ]
                data.columns = columns

                num_imputer = IterativeImputer(random_state=0)
                data[self.numerical_features] = num_imputer.fit_transform(data[self.numerical_features])

                cat_imputer = SimpleImputer(strategy='most_frequent')
                data[self.categorical_features] = cat_imputer.fit_transform(data[self.categorical_features])

                data['disease'] = data['disease'].apply(lambda x: 1 if x > 0 else 0)
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
        self.train_data, self.test_data = train_test_split(
            combined_data, test_size=0.2, random_state=42, stratify=combined_data['disease']
        )

    def preprocess_data(self, data, fit=True):
        if fit:
            num_pipelines = []
            for feature in self.numerical_features:
                pipeline = Pipeline([
                    ('selector', ColumnSelector(columns=[feature])),
                    ('scaler', StandardScaler()),
                    ('weigher', FeatureWeigher(weight=self.weights[feature]))
                ])
                num_pipelines.append((feature, pipeline, [feature]))

            cat_pipelines = []
            for feature in self.categorical_features:
                pipeline = Pipeline([
                    ('selector', ColumnSelector(columns=[feature])),
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                    ('weigher', FeatureWeigher(weight=self.weights[feature]))
                ])
                cat_pipelines.append((feature, pipeline, [feature]))

            self.preprocessor = ColumnTransformer(transformers=num_pipelines + cat_pipelines)
            processed_data = self.preprocessor.fit_transform(data)
        else:
            processed_data = self.preprocessor.transform(data)

        return processed_data

    def build_nn_model(self, input_dim):
        model = Sequential()
        model.add(tf.keras.layers.InputLayer(shape=(input_dim,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train_nn_model(self):
        X = self.train_data.drop(['disease'], axis=1)
        y = self.train_data['disease']

        X_processed = self.preprocess_data(X)
        input_dim = X_processed.shape[1]

        self.nn_model = self.build_nn_model(input_dim)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        self.nn_model.fit(
            X_processed, y, epochs=100, batch_size=32, verbose=0,
            validation_split=0.2, callbacks=[early_stopping]
        )

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

        y_pred_prob = self.nn_model.predict(X_test_processed, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        print("Result:")
        print(classification_report(y_test, y_pred))

    def predict(self, X):
        X_processed = self.preprocess_data(X, fit=False)
        if isinstance(X_processed, np.ndarray) and X_processed.ndim == 1:
            X_processed = X_processed.reshape(1, -1)
        y_prob = self.nn_model.predict(X_processed, verbose=0)
        y = (y_prob > 0.5).astype(int).flatten()
        return y

    def calculate_correlation(self, x1, x2):
        if x1 in self.train_data.columns and x2 in self.train_data.columns:
            correlation = self.train_data[x1].corr(self.train_data[x2])
            print(f"Correlation between {x1} and {x2}: {correlation}")
            return correlation

        print(f"[ERR] One or both features '{x1}' and '{x2}' not found.")
        return None

    def set_weight(self, feature, weight):
        if feature in self.weights:
            self.weights[feature] = weight
            print(f"Weight for feature '{feature}' set to {weight}.")
            self.train_nn_model()
        else:
            print(f"[ERR] Feature '{feature}' not found.")

def main():
    start_time = time.time()
    predictor = HeartDiseasePredictor()
    # first_test_sample = predictor.test_data.iloc[[0]].drop(['disease'], axis=1)
    # y = predictor.predict(first_test_sample)
    # print(f"Prediction for the first test sample: {y[0]} - Actual: {predictor.test_data.iloc[0]['disease']}")

    # x1, x2 = 'age', 'disease'
    # predictor.calculate_correlation(x1, x2)

    # # Beispiel für das Setzen eines Gewichts
    # predictor.set_weight('age', 10.0)
    # first_test_sample = predictor.test_data.iloc[[0]].drop(['disease'], axis=1)
    # y = predictor.predict(first_test_sample)
    # print(f"Prediction for the first test sample: {y[0]} - Actual: {predictor.test_data.iloc[0]['disease']}")

    print(f"Execution time: {time.time() - start_time}")

if __name__ == "__main__":
    tf.config.optimizer.set_jit(False)
    main()
