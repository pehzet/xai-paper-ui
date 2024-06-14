import streamlit as st    
from icecream import ic
def get_assistant_response():
    msg = st.session_state.assistant.messages[-1]
    response = st.session_state.assistant.chat(msg["content"])
    return response
def chat_page(): 
    st.markdown("<h1 style='text-align: center;'>XAI Bot</h1>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        # st.session_state.assistant = XAIAssistant()
        st.session_state.messages = st.session_state.assistant.messages

    # Display chat messages
    for msg in st.session_state.assistant.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Wie kann ich helfen?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.assistant.messages.append({"role": "user", "content": prompt})
        # Get assistant's response
        with st.spinner("Bin gleich wieder da..."):
            response = get_assistant_response()
        # Display assistant's response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)