import streamlit as st    
from icecream import ic
# from chatbot import XAIChatbot
import base64
from PIL import Image
from io import BytesIO

def render_image(base64_string=None):
    if not base64_string:
        with open("base64_img_test.txt", "r") as f:
            base64_string = f.read()
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))
    return img
def get_assistant_response(msg):
    response = st.session_state.assistant.chat(msg)
    # response = st.session_state.assistant.chat(msg["content"])
    img = render_image(None)
    return response, img
def get_messages():
    msgs = [msg for msg in st.session_state.assistant.messages[2:] if msg["role"] in ["user","assistant"]]
    # msgs = [{"role": "assistant", "content":"WOLOLO" },{"role": "user", "content":"AAAAH"}, {"role": "assistant", "content":"WOLOLO" }]
    return msgs
def chat_page(): 

    msgs = get_messages()
    for msg in msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Wie kann ich helfen?"):
   
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Bin gleich wieder da..."):
            response, img = get_assistant_response(prompt)
        # Display assistant's response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
            if img:
                st.image(img)