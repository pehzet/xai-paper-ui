import streamlit as st    
from icecream import ic
# from chatbot import XAIChatbot
import base64
from PIL import Image
from io import BytesIO
import re
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
def is_base64_image(data: str) -> bool:
    """
    Überprüft, ob der gegebene String ein Base64-enkodiertes Bild ist.
    """
    try:
        xy = base64.b64decode(data)
        return True
    except Exception:
        return False
def _remove_images_from_text(text):
    """
    Entfernt Bilder im Markdown-Stil aus dem Text, einschließlich Base64-Bilder und Bild-URLs.

    :param text: Der Originaltext, der bereinigt werden soll.
    :return: Der bereinigte Text ohne Bilder.
    """
    cleaned_text = re.sub(r"!\[.*?\]\(data:image/png;base64,.*?\)", "", text, flags=re.DOTALL)

    print(cleaned_text)
    
    return cleaned_text
def get_messages():
    messages = st.session_state.assistant.get_messages()
    extracted_messages = []

    # Nachrichten durchgehen, beginnend nach msg2
    for message in messages[2:]:
        role = message.get("role")
        content = message.get("content")
        is_img = False

        # Ignoriere Nachrichten mit None oder technische Angaben
        if content is None or (isinstance(content, str) and content.startswith("{") and content.endswith("}")):
            continue

        # Überprüfen, ob es sich um eine user- oder assistant-Nachricht handelt
        if role in ["user", "assistant"]:

            extracted_messages.append({
                "role": role,
                "content": _remove_images_from_text(content),
                "is_img": is_img
            })

        # Prüfe, ob es sich bei Tool-Nachrichten um ein Bild handelt
        elif role == "tool" and content is not None:
            is_img = is_base64_image(content)
            if is_img:
                extracted_messages.append({
                    "role": "assistant",
                    "content": content,
                    "is_img": is_img
                })
    

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
            if img_base64:
                img = render_image(img_base64)
                st.image(img)
            st.markdown(_remove_images_from_text(response))
           

            
