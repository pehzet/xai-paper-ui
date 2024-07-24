from xai_assistant_cls import XAIAssistant
from icecream import ic
import streamlit as st
import os
import datetime
import uuid
from utils import clean_latex_formatting, save_state_json, get_tipi_result_from_session_state 
from welcome import welcome_page
from survey import survey
from chat import chat_page
from images import show_images
from decision import decision
from thanks import thank_you_page
from tipi import get_description_from_scores
def set_page_styling():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
    }

    .stTextInput, .stNumberInput, .stSelectSlider {
        margin-bottom: 20px;
    }

    .stTextInput input, .stNumberInput input {
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 4px;
    }

    .stSelectSlider {
        padding: 10px 0;
    }
    .StyledThumbValue.st-emotion-cache-132r2mp.ew7r33m2 {
        font-family: 'Roboto', sans-serif;
        color: darkgrey;
    }
    .st-emotion-cache-1inwz65.ew7r33m0 {
        font-family: 'Roboto', sans-serif;
        color: darkgrey;
    }

    .st-emotion-cache-1s3l9q9.e1nzilvr5 {
        font-family: 'Roboto', sans-serif;
        color: black;
    }


    .stButton button:hover {
        background-color: #45a049;
    }

    h1 {
        font-weight: 700;
    }

    label {
        font-weight: 400;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize():
    set_page_styling()
    # assistant = XAIAssistant()
    user_uuid = str(uuid.uuid4())
    start_time = datetime.datetime.now().isoformat()
    if not "assistant" in st.session_state:
        if os.environ.get("SMOKE_TEST"):
            ic("Smoke test detected. Setting assistant to None.")
            st.session_state["assistant"] = None
        else:
            from config import ASSISTANT_ID
            st.session_state["assistant"] = XAIAssistant(assistant_id=ASSISTANT_ID)
    st.session_state["user_uuid"] = user_uuid
    st.session_state["start_time"] = start_time
    st.session_state["finished"] = False
    save_state_json()
# later move to LLM -> currently circular import
def update_instructions():
    if "tipi_scores" not in st.session_state:
        return
    if "instructions_updated" in st.session_state:
        return
    tipi_scores = get_tipi_result_from_session_state(st.session_state["user_uuid"])
    
    instruction_additions = {
        "[[TIPI]]" : get_description_from_scores(tipi_scores)
    }
    instructions = st.session_state["assistant"].prepare_instructions(instruction_additions)
    st.session_state["assistant"].update_instructions(instructions)
    st.session_state["instructions_updated"] = True
def main():
    initialize()
    if "page" not in st.session_state:
        st.session_state["page"] = "welcome"

    if st.session_state["page"] == "welcome":
        welcome_page()

    elif st.session_state["page"] == "survey":
        survey()
        save_state_json()
    elif st.session_state["page"] == "chat":
        update_instructions()
        show_images()
        decision()
        chat_page()
        save_state_json()
    elif st.session_state["page"] == "thanks":
        st.session_state["finished"] = True
        save_state_json()
        thank_you_page()

if __name__ == "__main__":
    main()