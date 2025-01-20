import streamlit as st    
from icecream import ic
# from chatbot import XAIChatbot
import base64
from PIL import Image
from io import BytesIO

def render_image(base64_string=None):
    if not base64_string:
        return None
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))
    return img
def get_assistant_response(msg):
    response, img = st.session_state.assistant.chat(msg)
    # response = st.session_state.assistant.chat(msg["content"])
    # img = render_image(None)
    return response, img


def get_messages():
    # TODO MESSAGES ARE EMPTY - FIX
    messages = st.session_state.assistant.get_messages()
    ic(messages)
    extracted_messages = []

    for message in messages[2:]:
        role = message['role']
        if role == 'system':
            continue
        content = message['content']
        is_img = False

        # Check if the message contains a tool call
        if 'tool_calls' in message and message['tool_calls']:
            for tool_call in message['tool_calls']:
                if tool_call['function']['name'] == 'generate_shap_diagram':
                    is_img = True
                    content = message['content']  # Retain the original base64 image content

        extracted_messages.append({
            "role": role,
            "content": content,
            "is_img": is_img
        })
    ic(extracted_messages)
    return extracted_messages



def chat_page(): 

    msgs = get_messages()
    for msg in msgs:
        with st.chat_message(msg["role"]):
            if msg["is_img"]:
                img = render_image(msg["content"])
                st.image(img)
            else:
                st.markdown(msg["content"])
    
    if prompt := st.chat_input("Wie kann ich helfen?"):
   
        with st.chat_message("user"):
            st.markdown(prompt)


        with st.spinner("Bin gleich wieder da..."):
            response, img_base64 = get_assistant_response(prompt)
        # Display assistant's response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
           
            if img_base64:
                img = render_image(img_base64)
                st.image(img)
            
