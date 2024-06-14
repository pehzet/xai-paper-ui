import streamlit as st
from tipi import calculate_scores
from icecream import ic
def submit_survey():
    st.session_state["survey_completed"] = True
    st.session_state["tipi_scores"] = calculate_scores(st.session_state)
    st.session_state["page"] = "chat"


def survey():
    st.session_state["survey"] = {}
    st.title("Survey")
    st.write("Please fill out the survey below:")
    
    # Likert scale options
    options = [
        "Disagree strongly",
        "Disagree moderately",
        "Disagree a little",
        "Neither agree nor disagree",
        "Agree a little",
        "Agree moderately",
        "Agree strongly"
    ]


    questions = [
        "I see myself as extraverted, enthusiastic.",
        "I see myself as critical, quarrelsome.",
        "I see myself as dependable, self-disciplined.",
        "I see myself as anxious, easily upset.",
        "I see myself as open to new experiences, complex.",
        "I see myself as reserved, quiet.",
        "I see myself as sympathetic, warm.",
        "I see myself as disorganized, careless.",
        "I see myself as calm, emotionally stable.",
        "I see myself as conventional, uncreative."
    ]
    with st.form(key='my_form'):
        name = st.text_input("Name", value=st.session_state.get("name", ""))
        age = st.number_input("Age", value=st.session_state.get("age", 18), min_value=0, max_value=100)

        for i, q in enumerate(questions):
            q_key = f"q{i+1}"
            if q_key not in st.session_state:
                st.session_state["survey"][q_key] = options[0] #"Neither agree nor disagree"

        for j, question in enumerate(questions):
            q_key = f"q{j+1}"
            response = st.select_slider(question, options=options, key=question, value=None)
            st.session_state[q_key] = response
        st.session_state["survey"]["name"] = name
        st.session_state["survey"]["age"] = age

        submit_button = st.form_submit_button("Submit", on_click=submit_survey)


# https://www.truity.com/test/big-five-personality-test
# kurze Form: https://gosling.psy.utexas.edu/wp-content/uploads/2014/09/JRP-03-tipi.pdf