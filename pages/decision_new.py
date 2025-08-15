from .chat_page import chat_page
from .image_page import show_images
from .decision import test_case_table, decision_dropdown, init_new_decision
from .data_table import show_data_table
import streamlit as st
from datetime import datetime
import random



def prediction_element():
    st.markdown("The Neural Network predicts with an **accuracy of 85 %**, based on the data above, that the best crop to plant is:")
    st.markdown(f"<p style='font-size: 30px;'> {st.session_state.prediction}</p>", unsafe_allow_html=True)
def decision_new():
    if not str(st.session_state.decision_no) in st.session_state.decision_times:
        st.session_state.decision_times[str(st.session_state.decision_no)] = {}
    if "start" not in st.session_state.decision_times[str(st.session_state.decision_no)]:
        st.session_state.decision_times[str(st.session_state.decision_no)]["start"] = datetime.now().isoformat()
    st.progress(0.25 * int(st.session_state.decision_no))
    st.markdown(f"<p style='font-size: 20px;'> Decision {st.session_state.decision_no} of 4 </p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 20px;'> Your task is to decide which crop is the best to sow based on the given data in the table below and the feature importance diagram on the top right.<p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 20px;'> Select the crop from the dropdown list and click on the Submit button. <p>", unsafe_allow_html=True)
    row1 = st.container(border=True)
    with row1:
        st.markdown("Select the crop you think is best from the dropdown list below.")
        decision_dropdown()
    row2 = st.container(border=True)
    col1_1, col1_2 = st.columns(2, gap="small", border=True)
    with row2:
        with col1_1:
            test_case_table()
        with col1_2:
            show_images()
    # row2 = st.container(border=True)
    # with row2:
    #     prediction_element()
    row3 = st.container(border=True)

    col3_1, col3_2 = st.columns(2, gap="small", border=True)
    with row3:
        with col3_1:
            show_data_table()
        with col3_2:
            # st.markdown("**Cropify Decision Assistant** (scroll down to see the input field )")
            # chat_page()
            pass

    st.session_state.new_decision = False