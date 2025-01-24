import streamlit as st

def save_user_id_to_session_state(user_id):

    if "user_id" not in st.session_state:
        st.session_state["user_id"] = user_id
def welcome_page():
    st.title("Welcome to CropGPT")
    st.write("This is an introduction page. Click the button below to start the experiment.")

    user_id = st.text_input("Enter your Prolific ID")
    if st.button("Start Experiment"):
        save_user_id_to_session_state(user_id)
        st.session_state["page"] = "chat"
        st.rerun()

