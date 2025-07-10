import streamlit as st
import time

import pandas as pd
import os
from datetime import datetime
from prediction_model.model_interface import predict, predict_probabilities
import random
import math
from icecream import ic
FEATURE_METADATA = {
    "N": {"unit": "kg/ha", "min": 0, "max": 140, "info": "Nitrogen content in the fertilizer (kg/ha)"},
    "P": {"unit": "kg/ha", "min": 5, "max": 145, "info": "Phosphorus content in the fertilizer (kg/ha)"},
    "K": {"unit": "kg/ha", "min": 5, "max": 205, "info": "Potassium content in the fertilizer (kg/ha)"},
    "temperature": {"unit": "°C", "min": 8.83, "max": 41.95, "info": "Mean Temperature (°C) of the region in a year"},
    "humidity": {"unit": "%", "min": 14.26, "max": 94.96, "info": "Mean relative humidity (%) of the region in a year"},
    "ph": {"unit": "pH", "min": 3.50, "max": 9.94, "info": "Current Soil pH value"},
    "rainfall": {"unit": "mm/month", "min": 5.31, "max": 298.56, "info": "Average monthly rainfall (mm) of the region"},
}

def get_decision_id():
    if st.session_state.decision_no <= 3:
        decision_id = st.session_state.decision_no 
    else:
        while True:
            decision_id = random.randint(4, 10)
            if decision_id not in st.session_state.done_decision_ids:
                break
    return decision_id
def get_test_case(decision_id):

    base_dir = os.path.dirname(os.path.abspath(__file__))  # aktuelles Verzeichnis
    parent_dir = os.path.dirname(base_dir)
    test_cases_file = os.path.join(parent_dir,"prediction_model", "data", "test_cases.csv")
    test_cases = pd.read_csv(test_cases_file)
    test_case = test_cases.iloc[decision_id - 1]
    st.session_state["true_label"] = test_case["label"]
   
    test_case = test_case.drop('label')

    return test_case.to_dict()
def get_test_case_with_metadata(test_case):
    """Erweitert den Test-Case um statische Informationen (Einheit, Min, Max)."""
    

    # Erstelle eine erweiterte Tabelle mit zusätzlichen Informationen
    test_case_data = []
    for feature, value in test_case.items():
        if feature in FEATURE_METADATA:
            metadata = FEATURE_METADATA[feature]

            test_case_data.append({
                "Feature": feature,
                "Current Value": value,
                "Unit": metadata["unit"],
                "Min": metadata["min"],
                "Max": metadata["max"],
                "Info": metadata["info"]
            })
    df = pd.DataFrame(test_case_data).set_index("Feature")
    df = df.round(2)
    return df
def test_case_table():
    if st.session_state.new_decision:
        decision_id = get_decision_id()
        st.session_state.decision_id = decision_id
        
      
        prediction = predict(st.session_state.test_case)
 
        st.session_state.prediction = prediction
        st.session_state.predicted_labels.append(prediction)
    test_case_df = get_test_case_with_metadata(st.session_state.test_case)
    st.dataframe(test_case_df)


def get_decision_id():
    if st.session_state.decision_no <= 3:
        decision_id = st.session_state.decision_no 
    else:
        while True:
            decision_id = random.randint(4, 10)
            if decision_id not in st.session_state.done_decision_ids:
                break
    return decision_id

def decision_dropdown():
    decision_id = get_decision_id()
    st.session_state.test_case = get_test_case(decision_id)
    probs = predict_probabilities(st.session_state.test_case)

    # options = [("rice", 0.01), ("soybeans", 0.08), ("banana", 0.03), ("beans", 0.04), ("cowpeas", 0.05), ("orange", 0.06),
    #           ("maize", 0.07), ("coffee", 0.08), ("peas", 0.09), ("groundnuts", 0.10), ("mango", 0.11),
    #           ("watermelon", 0.12), ("grapes", 0.13), ("apple", 0.14), ("cotton", 0.15)]
    options = [(crop, probs.get(crop, 0)) for crop in probs.keys()]

    # options_for_display = [f"{crop} ({math.ceil(prob*100)} %)" for crop, prob in options]
    options_for_display = [f"{crop}" for crop, prob in options]
    # sort by name
    options_for_display.sort()

    selection = st.selectbox(
        "Selection",
        options_for_display,
        index=None,
        placeholder="Select the crop to plant",
        label_visibility="collapsed"
    )


    if st.button("Submit"):
        if selection is None:
            st.error("Please select a crop to plant.")

        else:
            print("Correct Choices:", st.session_state.correct_choices)
            decision = selection.split(" (")[0]
            print(decision)
            st.session_state["choices"][st.session_state.decision_no] = decision
            st.session_state.decision_made = True

def decision():
    if st.session_state.new_decision:
        decision_id = get_decision_id()
        st.session_state.test_case = get_test_case(decision_id)
        st.session_state.prediction = predict(st.session_state.test_case)
    st.write(f"**Decision {st.session_state.decision_no}**")
    st.write("Task: Select the crop to plant based on the given data in the table below. Use the prediction to help you decide.")
    # st.write(f"Prediction of the Neural Network (85 % Accurancy): **{st.session_state.prediction}**")
    test_case_df = get_test_case_with_metadata(st.session_state.test_case)
    st.table(test_case_df)
    
    st.session_state.new_decision = False

    options = ['rice', 'soybeans', 'banana', 'beans', 'cowpeas', 'orange', 'maize', 'coffee', 'peas', 'groundnuts', 'mango', 'watermelon', 'grapes', 'apple', 'cotton']
    
    decision = st.selectbox("", options, placeholder="Select the crop to plant", index=None)
    submit = st.button("Submit")
    if submit:
        st.session_state["choices"][st.session_state.decision_no] = decision
        decision = None
        # st.session_state.decision_no += 1
        st.session_state.decision_made = True

    
