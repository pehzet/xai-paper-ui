import streamlit as st
import os
import json
from webdav3.client import Client

title = "XAI Paper"
st.set_page_config(layout="wide", page_title=title, initial_sidebar_state="collapsed")

from pages.welcome import welcome_page
from pages.image_page import show_images
from pages.decision import decision
from pages.thanks import thanks
from pages.chat_page import chat_page
from pages.decision_new import decision_new
from pages.survey import show_survey
from chatbot import XAIChatbot

import copy
import json
from datetime import datetime
import uuid

def init():
    if "assistant" not in st.session_state:
        st.session_state.assistant = XAIChatbot(decision_no=1)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    if "decision_no" not in st.session_state:
        st.session_state.decision_no = 1
    if "decision_id" not in st.session_state:
        st.session_state.decision_id = 1
    if "done_decision_ids" not in st.session_state:
        st.session_state.done_decision_ids = []
    if "predicted_labels" not in st.session_state:
        st.session_state.predicted_labels = []
    if "decision_made" not in st.session_state:
        st.session_state.decision_made = False
    if "new_decision" not in st.session_state:
        st.session_state.new_decision = True
    if "choices" not in st.session_state:
        st.session_state.choices = {}
    if "decision_times" not in st.session_state:
        st.session_state.decision_times = {}
    if "survey_completed" not in st.session_state:
        st.session_state.survey_completed = False
    if not "experiment_start" in st.session_state:
        st.session_state.experiment_start = datetime.now().isoformat()


def save_session_state():
    user_id = st.session_state.get("user_id", None)
    if user_id is None:
        print("User ID not found. Generating UUID.")
        user_id = str(uuid.uuid4())
    session_state_copy = copy.deepcopy(st.session_state)
    session_state_copy.pop("assistant")
    session_state_dict = {k: v for k, v in session_state_copy.items()}

    with open(f"session_state_{user_id}.json", "w", encoding="utf-8") as f:
        json.dump(session_state_dict, f)
    # upload_session_state(user_id)


def upload_session_state(user_id):
    try:
        sciebo_config = st.secrets.get("sciebo")
        if sciebo_config is None:
            raise ValueError("Sciebo config not found.")
        filename = f"session_state_{user_id}.json"
        filepath = f"session_state_{user_id}.json"

        options = {
            'webdav_hostname': sciebo_config.get('SCIEBO_URL'),
            'webdav_login': sciebo_config.get('SCIEBO_LOGIN'),
            'webdav_password': sciebo_config.get('SCIEBO_PASSWORD')
        }
        client = Client(options)
        
        remote_path = os.path.join(sciebo_config.get('SCIEBO_DIRECTORY', ''), filename).replace('\\', '/')
        
        client.upload_file(
            remote_path=remote_path,
            local_path=filepath
        )
        
        print(f"File successfully uploaded to Sciebo: {remote_path}")
        
    except Exception as e:
        print(f"Error saving/uploading results: {str(e)}")



def close_decision():
    st.session_state.decision_times[str(st.session_state.decision_no)]["end"] =  datetime.now().isoformat()
    st.session_state.decision_made = False
    st.session_state.new_decision = True
    st.session_state.chat_history[st.session_state.decision_no] = st.session_state.assistant.get_messages()
    st.session_state["page"] = "survey"
    st.session_state.survey_completed = False
    save_session_state()
    st.rerun()


def complete_survey():
    
    st.session_state.decision_no += 1
    if st.session_state.decision_no > 10:
        st.session_state["page"] = "thanks"
    else:
        st.session_state["page"] = "chat"
    st.session_state.assistant = XAIChatbot(decision_no=st.session_state.decision_no)
    save_session_state()
    st.session_state.survey_completed = False
    st.rerun()


def main():
    if "page" not in st.session_state:
        st.session_state["page"] = "welcome"
    
    if st.session_state["page"] == "welcome":
        welcome_page()
    elif st.session_state["page"] == "chat":
        if st.session_state.new_decision:
            st.session_state.assistant = XAIChatbot(decision_no=st.session_state.decision_no)
        decision_new()
        if st.session_state.decision_made:
            close_decision()
    elif st.session_state["page"] == "survey":
        show_survey()
        if st.session_state.survey_completed:
            complete_survey()
    elif st.session_state["page"] == "thanks":
        st.session_state.experiment_end = datetime.now().isoformat()
        thanks()


if __name__ == "__main__":
    init()
    main()
