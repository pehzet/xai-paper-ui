import streamlit as st
from PIL import Image
import os
def show_images():
    # format_images_html()
    st.write("For further support you can view the global and local SHAP explanations for the decision.")
    img1_pth = f"case{st.session_state.decision_no}_global.png"
    img2_pth = f"case{st.session_state.decision_no}_local.png"
    base_dir = os.path.dirname(os.path.abspath(__file__))  # aktuelles Verzeichnis
    parent_dir = os.path.dirname(base_dir)
    image1_path = os.path.join(parent_dir, "images", img1_pth)
    image2_path = os.path.join(parent_dir, "images", img2_pth)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image1_path, caption="Global Explanation", use_container_width=True)
        
    with col2:
        st.image(image2_path, caption="Local Explanation", use_container_width=True)