import streamlit as st
def welcome_page():
    st.title("Welcome to our App")
    st.write("This is an introduction page. Click the button below to start the survey.")
    if st.button("Start Survey"):
        st.session_state["page"] = "survey"
        st.rerun()