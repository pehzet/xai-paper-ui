import streamlit as st
from pages.welcome import welcome_page
from pages.images import show_images
from pages.decision import decision
from pages.thanks import thanks
from pages.chat_page import chat_page
from chatbot import XAIChatbot

def init():
    if not "assistant" in st.session_state:
        st.session_state.assistant = XAIChatbot()
    title = "XAI Paper"
    st.set_page_config(
        page_title=title,
        initial_sidebar_state="collapsed",
        
)

def main():
   
    if "page" not in st.session_state:
        st.session_state["page"] = "welcome"

    if st.session_state["page"] == "welcome":
        welcome_page()
    elif st.session_state["page"] == "chat":
        show_images()
        decision()
        chat_page()
    elif st.session_state["page"] == "thanks":
        thanks()
 

if __name__ == "__main__":
    init()
    main()