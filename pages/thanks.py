import streamlit as st
import os
import json
import pandas as pd
from icecream import ic
# def calculate_correct_choices():
#     #C:\code\xai-paper-new\prediction_model\data\test_cases.csv
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     parent_dir = os.path.dirname(current_dir)
#     test_cases_file = os.path.join(parent_dir, "prediction_model", "data", "test_cases.csv")
#     test_cases = pd.read_csv(test_cases_file)
#     ic(test_cases)
#     choices = st.session_state.choices
#     ic(choices)
#     done_decision_ids = st.session_state.done_decision_ids
#     correct_choices = 0
#     for decision_idx, choice in choices.items():
#         ic(decision_idx)
#         decision_idx = int(decision_idx)
#         decision_id = done_decision_ids[decision_idx- 1]
#         ic(decision_id)
#         test_case = test_cases.iloc[decision_id - 1]
#         ic(test_case)
#         if choice == test_case["label"]:
#             correct_choices += 1
#     return correct_choices
def thanks():
    st.title("Thank You Page")

    st.write(f"You made **{st.session_state.correct_choices}/{len(st.session_state.choices)}** correct choices.")
    st.write("You can close the tab now and return to the survey.")

