import re
import json
import streamlit as st
import datetime
import os
from icecream import ic
def clean_latex_formatting(text: str) -> str:
    # Entfernt alle LaTeX-Mathematik-Umgebungen
    cleaned_text = re.sub(r"\\\(", "", text)
    cleaned_text = re.sub(r"\\\)", "", cleaned_text)
    return cleaned_text

def save_state_json():
    uuid = st.session_state.get("user_uuid")
    session_state_dict = {}
    for key, value in st.session_state.items():
        if key not in ["assistant"]:
            session_state_dict[key] = str(value)
    session_state_dict["last_updated"] = datetime.datetime.now().isoformat()

    if not os.path.exists("state_data"):
        os.makedirs("state_data")
    with open(f"state_data\{uuid}.json", "w") as f:
        json.dump(session_state_dict, f)

def get_tipi_result_from_session_state(uuid):
    with open(f"state_data\{uuid}.json", "r") as f:
        session_state_dict = json.load(f)

    return session_state_dict["tipi_scores"]
# def get_instructions():
#     with open("instructions.txt", "r") as f:
#         instructions = f.read()
#     return instructions

# def get_instructions_with_additions(instruction_additions):
#         instructions = get_instructions()

#         for placeholder, instruction_addition in instruction_additions.items():
#             position = instructions.find(placeholder)
#             instructions = instructions[:position] + instruction_addition + instructions[position:]