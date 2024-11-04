import streamlit as st
def welcome_page():
    st.title("Welcome to CropGPT")
    st.write("This is an introduction page. Click the button below to start the experiment.")

    if st.button("Start Experiment"):
        st.session_state["page"] = "chat"
        st.rerun()
        