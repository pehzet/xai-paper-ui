import streamlit as st
from survey import survey
from chat import chat
st.title("Explain me the explainable")
if "survey_completed" not in st.session_state:
    st.session_state["survey_completed"] = False
if not st.session_state.survey_completed:
    survey()
else:
    chat()