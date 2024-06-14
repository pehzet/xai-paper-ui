import streamlit as st
import time
from icecream import ic
def decision():
    # TODO: Format that buttons are "more beautiful"
    # Create two columns
    col1, col2 = st.columns(2)

    # Check if 'button_clicked' is in session_state, if not, initialize it to None
    if 'button_clicked' not in st.session_state:
        st.session_state["button_clicked"] = None

    # Create buttons in each column
    with col1:
        btn1 = st.button("Maintain", on_click=lambda: st.session_state.update(button_clicked="Maintain"))

    with col2:
        btn2 = st.button("Not Maintain", on_click=lambda: st.session_state.update(button_clicked="Not Maintain"))

    if st.session_state["button_clicked"]:
        st.write(f"You decided for: {st.session_state['button_clicked']}")
        time.sleep(1)
        st.session_state["page"] = "thanks"
        # had to add rerun() to make the page change. Otherwise it only worked after clicking a button twice -> TODO: find out why and fix
        st.rerun()
