import streamlit as st
import time
from icecream import ic
import pandas as pd
import os
from datetime import datetime
from prediction_model.model_interface import predict
def get_test_case():

    st.session_state.decision_times[str(st.session_state.decision_no)] = {}
    st.session_state.decision_times[str(st.session_state.decision_no)]["start"] = datetime.now().isoformat()
    base_dir = os.path.dirname(os.path.abspath(__file__))  # aktuelles Verzeichnis
    parent_dir = os.path.dirname(base_dir)
    test_cases_file = os.path.join(parent_dir,"prediction_model", "data", "test_cases.csv")
    test_cases = pd.read_csv(test_cases_file)
    test_case = test_cases.iloc[st.session_state.decision_no-1]
    test_case = test_case.drop('label')

    return test_case.to_dict()

def decision():
    
    if st.session_state.new_decision:
        st.session_state.test_case = get_test_case()
        st.session_state.prediction = predict(st.session_state.test_case)
      
    st.table(st.session_state.test_case)
    st.write(f"Prediction: {st.session_state.prediction}")
    st.session_state.new_decision = False
    # Check if 'button_clicked' is in session_state, if not, initialize it to None

    options = ['rice', 'Soyabeans', 'banana', 'beans', 'cowpeas', 'orange', 'maize', 'coffee', 'peas', 'groundnuts', 'mango', 'watermelon', 'grapes', 'apple', 'cotton']
    # Create buttons in each column
    decision = st.selectbox("Was würdest du aussäen?",options, placeholder="Bitte wählen", index=None)
    submit = st.button("Submit")
    if submit:
        st.session_state["choices"][st.session_state.decision_no] = decision
        # st.session_state.decision_no += 1
        st.session_state.decision_made = True
        # if st.session_state.decision_no > 10    :
        #     st.session_state["page"] = "thanks"
        # st.rerun()
    
