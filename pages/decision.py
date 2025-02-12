import streamlit as st
import time

import pandas as pd
import os
from datetime import datetime
from prediction_model.model_interface import predict
import random
FEATURE_METADATA = {
    "N": {"unit": "kg/ha", "min": 0, "max": 140},
    "P": {"unit": "kg/ha", "min": 5, "max": 145},
    "K": {"unit": "kg/ha", "min": 5, "max": 205},
    "temperature": {"unit": "°C", "min": 8.83, "max": 41.95},
    "humidity": {"unit": "%", "min": 14.26, "max": 94.96},
    "ph": {"unit": "pH", "min": 3.50, "max": 9.94},
    "rainfall": {"unit": "mm/month", "min": 5.31, "max": 298.56},
}

def get_decision_id():
    print(st.session_state.decision_no)
    if st.session_state.decision_no <= 3:
        decision_id = st.session_state.decision_no 
    else:
        while True:
            decision_id = random.randint(4, 10)
            if decision_id not in st.session_state.done_decision_ids:
                break
    return decision_id
def get_test_case(decision_id):

    st.session_state.done_decision_ids.append(decision_id)
    st.session_state.decision_times[str(st.session_state.decision_no)] = {}
    st.session_state.decision_times[str(st.session_state.decision_no)]["decision_id"] = decision_id
    st.session_state.decision_times[str(st.session_state.decision_no)]["start"] = datetime.now().isoformat()
    base_dir = os.path.dirname(os.path.abspath(__file__))  # aktuelles Verzeichnis
    parent_dir = os.path.dirname(base_dir)
    test_cases_file = os.path.join(parent_dir,"prediction_model", "data", "test_cases.csv")
    test_cases = pd.read_csv(test_cases_file)
    test_case = test_cases.iloc[decision_id - 1]
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
                "Max": metadata["max"]
            })

    return pd.DataFrame(test_case_data).set_index("Feature")
def decision():
    if st.session_state.new_decision:
        decision_id = get_decision_id()
        st.session_state.test_case = get_test_case(decision_id)
        st.session_state.prediction = predict(st.session_state.test_case)
    st.write(f"**Decision {st.session_state.decision_no}**")
    st.write("Task: Select the crop to plant based on the given data in the table below. Use the prediction to help you decide.")
    st.write(f"Prediction of the Neural Network (85 % Accurancy): **{st.session_state.prediction}**")
    test_case_df = get_test_case_with_metadata(st.session_state.test_case)
    st.table(test_case_df)
    
    st.session_state.new_decision = False
    # Check if 'button_clicked' is in session_state, if not, initialize it to None

    options = ['rice', 'Soyabeans', 'banana', 'beans', 'cowpeas', 'orange', 'maize', 'coffee', 'peas', 'groundnuts', 'mango', 'watermelon', 'grapes', 'apple', 'cotton']
    # Create buttons in each column
    decision = st.selectbox("Which crop would you plant?",options, placeholder="Please choose", index=None)
    submit = st.button("Submit")
    if submit:
        st.session_state["choices"][st.session_state.decision_no] = decision
        # st.session_state.decision_no += 1
        st.session_state.decision_made = True
        # if st.session_state.decision_no > 10    :
        #     st.session_state["page"] = "thanks"
        # st.rerun()
    
