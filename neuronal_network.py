import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from concurrent.futures import ThreadPoolExecutor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

REPEAT_RUNS = 50

class HeartDiseasePredictor:
    def __init__(self):
        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.nn_model = None
        self.preprocessor = None

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

                numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
                categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'ca']

                num_imputer = IterativeImputer(random_state=0)
                data[numerical_features] = num_imputer.fit_transform(data[numerical_features])

                cat_imputer = SimpleImputer(strategy='most_frequent')
                data[categorical_features] = cat_imputer.fit_transform(data[categorical_features])

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

    def build_nn_model(self, input_dim):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=input_dim))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def preprocess_data(self, data, fit=True):
        categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'ca']
        numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

        if fit:
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ]
            )
            processed_data = self.preprocessor.fit_transform(data)
        else:
            processed_data = self.preprocessor.transform(data)

        return processed_data

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
            X_processed, y, epochs=100, batch_size=32, verbose=1,
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

        y_pred_prob = self.nn_model.predict(X_test_processed)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        if REPEAT_RUNS == 1:
            print("Result:")
            print(classification_report(y_test, y_pred))

        return accuracy, f1

    def predict(self, X):
        X_processed = self.preprocess_data(X, fit=False)
        y_prob = self.nn_model.predict(X_processed)
        y = (y_prob > 0.5).astype(int).flatten()
        return y

    def calculate_correlation(self, x1, x2):
        if x1 in self.train_data.columns and x2 in self.train_data.columns:
            correlation = self.train_data[x1].corr(self.train_data[x2])
            print(f"Correlation between {x1} and {x2}: {correlation}")
            return correlation

        print(f"[ERR] One or both features '{x1}' and '{x2}' not found.")
        return None

def main(repeat_runs=0):
    predictor = HeartDiseasePredictor()

    times = []
    accuracies = []
    f1_scores = []

    if repeat_runs == 1:
        predictor.load_data_from_directory()
        predictor.train_nn_model()
        accuracy, f1 = predictor.run_nn_simulation()

        first_test_sample = predictor.test_data.iloc[[0]].drop(['disease'], axis=1)
        y = predictor.predict(first_test_sample)
        print(f"Prediction for the first test sample: {y[0]} - Actual: {predictor.test_data.iloc[0]['disease']}")

        x1, x2 = 'age', 'disease'
        predictor.calculate_correlation(x1, x2)

        predictor.train_nn_model()
        accuracy, f1 = predictor.run_nn_simulation()
    else:
        for _ in range(repeat_runs):
            start_time = time.time()
            predictor.load_data_from_directory()
            predictor.train_nn_model()
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
