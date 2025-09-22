import streamlit as st

def save_user_id_to_session_state(user_id):

    if "user_id" not in st.session_state:
        st.session_state["user_id"] = user_id
def welcome_page():
    st.title("Welcome to Cropify!")
    st.write("Enter your Prolific ID and click the button below to start the experiment.")

    user_id = st.text_input("Enter your Prolific ID")

    # Prüfen, ob Eingabe lang genug ist
    valid = len(user_id) >= 6

    if not valid:
        st.info("Your Prolific ID must be at least 6 characters long.")

    # Button wird deaktiviert, wenn Eingabe nicht gültig
    if st.button("Start Experiment", disabled=not valid):
        save_user_id_to_session_state(user_id)
        st.session_state["page"] = "chat"
        st.rerun()