import streamlit as st
import time

def show_wait():
    st.title("Training Phase Complete")
    
    st.success("🎉 Congratulations! You have completed the training phase.")
    st.info("The actual study will begin shortly...")
    
    # Wait 5 seconds then proceed
    time.sleep(5)
    st.session_state["page"] = "chat"
    st.rerun()
