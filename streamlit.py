import openai
import streamlit as st
import time
# Initialize the chat messages history
messages = [{"role": "assistant", "content": "How can I help?"}]
st.session_state["messages"] = messages
msgs = st.session_state["messages"]
# Function to display the chat history
def display_chat_history(messages):
    for message in messages:

        with st.chat_message(message['role'].capitalize()):
            st.write(message['content'])

# Function to get the assistant's response
def get_assistant_response(messages):
    # r = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[{"role": m["role"], "content": m["content"]} for m in messages],
    # )
    # response = r.choices[0].message.content
    response = "This is my message to you, uh uh"
    # msgs.append({"role": "assistant", "content": response})
    time.sleep(3)
    return response

# Main chat loop

st.title("💬 Chatbot")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
prompt = st.chat_input()
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.spinner("Thinking..."):
        response = get_assistant_response(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
    print(st.session_state.messages)