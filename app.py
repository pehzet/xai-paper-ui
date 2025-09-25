import streamlit as st
import os
import json
from webdav3.client import Client
import posixpath
title = "Cropify"
st.set_page_config(layout="wide", page_title=title, initial_sidebar_state="collapsed")

from pages.welcome import welcome_page
from pages.image_page import show_images
# from pages.decision import decision
from pages.thanks import thanks
from pages.chat_page import chat_page
from pages.decision_new import decision_new, init_new_decision
from pages.survey import show_survey
from pages.wait import show_wait
from pages.explain import show_explanation
from chatbot import XAIChatbot

import logging
import copy
import json
from datetime import datetime
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    if "decision_completed" not in st.session_state:
        st.session_state.decision_completed = False
    if "correct_choices" not in st.session_state:
        st.session_state.correct_choices = 0
    if "current_choice_is_correct" not in st.session_state:
        st.session_state.current_choice_is_correct = None
    if "page" not in st.session_state:
        st.session_state.page = "welcome"
    if "char_count" not in st.session_state:
        st.session_state.char_count = 0
    if not "experiment_start" in st.session_state:
        st.session_state.experiment_start = datetime.now().isoformat()
    if not "group_type" in st.session_state:
        st.session_state.group_type = "Invention/with-llm"


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
    upload_session_state(user_id)


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
        
        # remote_path = os.path.join(sciebo_config.get('SCIEBO_DIRECTORY', ''), filename).replace('\\', '/')
        remote_path = posixpath.join(sciebo_config.get('SCIEBO_DIRECTORY', ''), filename)
        
        client.upload_file(
            remote_path=remote_path,
            local_path=filepath
        )
        
        logger.info(f"File successfully uploaded to Sciebo: {remote_path}")
        
    except Exception as e:
        logger.error(f"Error saving/uploading results: {str(e)}")



def close_decision():

    
    st.session_state.done_decision_ids.append(st.session_state.decision_id)
    st.session_state.decision_times[str(st.session_state.decision_no)]["end"] =  datetime.now().isoformat()
    st.session_state.decision_made = False
    st.session_state.new_decision = True
    st.session_state.chat_history[st.session_state.decision_no] = st.session_state.assistant.get_messages()
    st.session_state["page"] = "survey"
    st.session_state.decision_completed = False
    st.session_state.char_count = 0


    save_session_state()
    st.rerun()


def complete_decision():

    st.session_state.decision_no += 1
    if st.session_state.decision_no > 4:
        st.session_state["page"] = "thanks"
    elif st.session_state.decision_no == 2:
        st.session_state["page"] = "wait"
    else:
        st.session_state["page"] = "chat"
    if st.session_state.current_choice_is_correct:
        st.session_state.correct_choices += 1
        st.session_state.current_choice_is_correct = None
    # st.session_state.assistant = XAIChatbot(decision_no=st.session_state.decision_no)
    save_session_state()
    st.session_state.decision_completed = False
    st.rerun()


def main():

    if st.session_state["page"] == "welcome":
        welcome_page()
    elif st.session_state["page"] == "chat":
        if st.session_state.new_decision:
            init_new_decision()
            st.session_state.assistant = XAIChatbot(decision_no=st.session_state.decision_id)

        decision_new()
        if st.session_state.decision_made:
            close_decision()
    elif st.session_state["page"] == "wait":
        show_wait()
    elif st.session_state["page"] == "survey":
        show_survey()

    elif st.session_state["page"] == "explain":
        show_explanation()
        if st.session_state.decision_completed:
            complete_decision()
    elif st.session_state["page"] == "thanks":
        st.session_state.experiment_end = datetime.now().isoformat()
        save_session_state()
        thanks()
        


if __name__ == "__main__":
    init()
    main()
