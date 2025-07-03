import streamlit as st    

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
        if not data.startswith(('data:image/', '/9j/', 'iVBORw0KGgo', 'R0lGODlh', 'UklGR')):
            return False
        
        if data.startswith('data:image/'):
            data = data.split(',', 1)[1]
        
        decoded = base64.b64decode(data, validate=True)
        
        img_headers = [
            b'\xff\xd8\xff',  # JPEG
            b'\x89PNG\r\n\x1a\n',  # PNG
            b'GIF87a',  # GIF87a
            b'GIF89a',  # GIF89a
            b'RIFF'  # WEBP (beginnt mit RIFF)
        ]
        
        return any(decoded.startswith(header) for header in img_headers)
        
    except Exception:
        return False
def _remove_images_from_text(text):
    """
    Entfernt Bilder im Markdown-Stil aus dem Text, einschließlich Base64-Bilder und Bild-URLs.

    :param text: Der Originaltext, der bereinigt werden soll.
    :return: Der bereinigte Text ohne Bilder.
    """
    cleaned_text = re.sub(r"!\[.*?\]\(data:image/png;base64,.*?\)", "", text, flags=re.DOTALL)


    
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
    # st.write("If you have any questions, feel free to ask the assistant.")
    # st.write("Decision Assistant")
    msgs = get_messages()
    chat_placeholder = st.empty()

    def render_chat(msgs, new_msg = None):
        with chat_placeholder.container(height=500, border=False): #height=250,
            with st.chat_message("assistant"):
                st.markdown("If you have any questions, feel free to ask me.")
            for msg in msgs:
                with st.chat_message(msg["role"]):
                    if msg["is_img"]:
                    
                        img = render_image(msg["content"])
                        st.image(img)
                    else:
                        st.markdown(msg["content"])
            if new_msg:
                with st.chat_message("user"):
                    st.markdown(new_msg)
    render_chat(msgs)
    if prompt := st.chat_input("I'm your Decision Assistant. How can I support you?"):
        st.session_state.char_count += len(prompt)

        # with st.chat_message("user"):
        #     st.markdown(prompt)
        render_chat(msgs, prompt)

        with st.spinner("Be right back..."):
            response, img_base64 = get_assistant_response(prompt)

        st.rerun()
           

            
