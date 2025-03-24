import streamlit as st
import os
import json
import pandas as pd
def calculate_correct_choices():
    #C:\code\xai-paper-new\prediction_model\data\test_cases.csv
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    test_cases_file = os.path.join(parent_dir, "prediction_model", "data", "test_cases.csv")
    test_cases = pd.read_csv(test_cases_file)
    choices = st.session_state.choices
    done_decision_ids = st.session_state.done_decision_ids
    correct_choices = 0
    for decision_idx, choice in choices.items():
        decision_idx = int(decision_idx)
        decision_id = done_decision_ids[decision_idx- 1]
        test_case = test_cases.iloc[decision_id - 1]
        if choice == test_case["label"]:
            correct_choices += 1
    return correct_choices
def thanks():
    st.title("Thank You Page")
    correct_choices = calculate_correct_choices()
    st.write(f"You made **{correct_choices}/{len(st.session_state.choices)}** correct choices.")
    st.write("Thank you for using our app.")

