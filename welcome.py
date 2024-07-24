import streamlit as st
def welcome_page():
    st.title("Welcome to our App")
    st.write("This is an introduction page. Click the button below to start the survey.")
    # st.write("Please enter your Name")
    # name = st.text_input("Name")
    # if name:
    #     st.session_state["user_name"] = name
    if st.button("Start Survey"):
        st.session_state["page"] = "survey"
        st.rerun()