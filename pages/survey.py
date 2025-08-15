import streamlit as st

# Inject custom CSS to change the slider color to #009ee3


def show_survey():

    st.title("Certainty Survey")

    questions = {
        "q1": "Please rate your level of certainty with your decision? (7-scale)",
    }
    responses = {}
    if not "survey" in st.session_state:
        st.session_state.survey = {}
    if not str(st.session_state.decision_no) in st.session_state.survey:
        st.session_state.survey[str(st.session_state.decision_no)] = {}
    labels = ["very low", "low", "somewhat low", "neutral", "somewhat high", "high", "very high"]
    labels_html = ["", "low", "somewhat low", "neutral", "somewhat high", "high", ""]
    for key, question in questions.items():
        st.markdown(
        """
        <style>
        .labels {
            display: flex;
            justify-content: space-between;
            padding: 0 12px;
            font-size: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
        st.session_state["survey"][str(st.session_state.decision_no)][key] = st.select_slider(question, options=labels, value="neutral")
        st.markdown(f"<div class='labels'>{''.join(f'<span>{t}</span>' for t in labels_html)}</div>", unsafe_allow_html=True)
    btn = st.button("Submit")
    if btn:
        # for key, value in responses.items():
        #     st.session_state[str(st.session_state.decision_no)]["survey"][key] = value

        st.session_state.survey_completed = True
        st.session_state["page"] = "explain"
        st.rerun()


