import streamlit as st
import random
import time
import json

def chat():
    def save_msgs_json():
        if not "file_name" in st.session_state:
            st.session_state["file_name"] = "chat_history.json"
        with open(st.session_state["file_name"], "w") as f:
            json.dump(st.session_state.messages, f)
    # Streamed response emulator
    def response_generator():
        response = random.choice(
            [
                "Philip S. is the best!",
                "Alena is the best!",
                "Philipp Z. is the best!",
            ]
        )
        for word in response.split():
            yield word + " "
            time.sleep(0.3)


    st.title("Explain me the explainable")
    col1, col2 = st.columns(2)
    col1.header("Figure 1")
    col1.image("image_1.png")
    col2.header("Figure 2")
    col2.image("image_2.png")


    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hello Human, what can i explain you today?"})
        print(st.session_state.messages)
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # st.markdown(message["content"])
            st.markdown(message["role"]).write(message["content"])

    # Accept user input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator())
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        save_msgs_json()