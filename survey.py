import streamlit as st

def submit_survey():
    st.session_state["survey_completed"] = True
def survey():
    st.title("Survey")
    st.write("Please fill out the survey below:")
    with st.form(key='my_form'):
        name = st.text_input("Name", value=st.session_state.get("name", ""))
        age = st.number_input("Age", value=st.session_state.get("age", 18), min_value=0, max_value=100)
        q1_answer = st.radio("Question 1: Do you like ice cream?", ["Yes", "No"])
        q2_answer = st.slider("Question 2: How much do you like ice cream?", min_value=0, max_value=10, step=1)
        st.session_state["name"] = name
        st.session_state["age"] = age
        st.session_state["q1_answer"] = q1_answer
        st.session_state["q2_answer"] = q2_answer
        submit_button = st.form_submit_button("Submit", on_click=submit_survey)


# https://www.truity.com/test/big-five-personality-test