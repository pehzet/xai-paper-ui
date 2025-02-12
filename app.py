import streamlit as st
from pages.welcome import welcome_page
from pages.images import show_images
from pages.decision import decision
from pages.thanks import thanks
from pages.chat_page import chat_page
from chatbot import XAIChatbot

import copy
import json
from datetime import datetime
import uuid
def init():
    if not "assistant" in st.session_state:
        st.session_state.assistant = XAIChatbot(decision_no=1)
    if not "chat_history" in st.session_state:
        st.session_state.chat_history = {}
    if not "decision_no" in st.session_state:
        st.session_state.decision_no = 1
    if not "decision_id" in st.session_state:
        st.session_state.decision_id = 1
    if not "done_decision_ids" in st.session_state:
        st.session_state.done_decision_ids = []
    if not "decision_made" in st.session_state:
        st.session_state.decision_made = False
    if not "new_decision" in st.session_state:
        st.session_state.new_decision = True
    if not "choices" in st.session_state:
        st.session_state.choices = {}
    if not "decision_times" in st.session_state:
        st.session_state.decision_times = {}
    st.session_state.experiment_start = datetime.now().isoformat()
    title = "XAI Paper"
    st.set_page_config(
        page_title=title,
        initial_sidebar_state="collapsed",     
)

def save_session_state():
    user_id = st.session_state.get("user_id", None)
    if user_id is None:
        print("User ID not found. Going to generate uuid.")
        user_id = str(uuid.uuid4())
    session_state_copy = copy.deepcopy(st.session_state)
    session_state_copy.pop("assistant")
    session_state_dict = {k: v for k, v in session_state_copy.items()}

    with open(f"session_state_{user_id}.json", "w", encoding="utf-8") as f:
        json.dump(session_state_dict, f)

def upload_session_state():
    #TODO: upload session state from json file to cloud
    pass

def close_decision():
    st.session_state.decision_times[str(st.session_state.decision_no)]["end"] = datetime.now().isoformat()
    #todo: remove increment in decision
    st.session_state.decision_made = False
    st.session_state.new_decision = True
    st.session_state["chat_history"][st.session_state.decision_no] = st.session_state.assistant.get_messages()

    st.session_state.decision_no += 1
    if st.session_state.decision_no > 10    :
        st.session_state["page"] = "thanks"
    st.session_state.assistant = XAIChatbot(decision_no=st.session_state.decision_no)
    save_session_state()
    st.rerun()
    

def main():
   
    if "page" not in st.session_state:
        st.session_state["page"] = "welcome"

    if st.session_state["page"] == "welcome":
        welcome_page()
    elif st.session_state["page"] == "chat":
        if st.session_state.new_decision:
            st.session_state.assistant = XAIChatbot(decision_no=st.session_state.decision_no)
            
        
        decision()
        show_images()
        chat_page()

        if st.session_state.decision_made:
            close_decision()
    elif st.session_state["page"] == "thanks":
        st.session_state.experiment_end = datetime.now().isoformat()
        thanks()
 

if __name__ == "__main__":
    init()
    main()