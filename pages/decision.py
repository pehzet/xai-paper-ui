import streamlit as st
import time
from icecream import ic
def decision():
    # TODO: Format that buttons are "more beautiful"


    # Check if 'button_clicked' is in session_state, if not, initialize it to None
    if 'choice' not in st.session_state:
        st.session_state["choice"] = None
    options = ["Bitte wählen", "Corn", "Wheat", "Rice", "Soybean"]
    # Create buttons in each column
    decision = st.selectbox("What Crop is it?",options)
    submit = st.button("Submit")
    if submit:
        st.session_state["choice"] = decision
        st.session_state["page"] = "thanks"
        st.rerun()
    
