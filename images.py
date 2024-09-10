import streamlit as st
from PIL import Image
import os
def show_images():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # aktuelles Verzeichnis
    image1_path = os.path.join(base_dir, "images", "image_1.png")
    image2_path = os.path.join(base_dir, "images", "image_2.png")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image1_path, caption="Global FI", use_column_width=True)
        
    with col2:
        st.image(image2_path, caption="Local Explanation", use_column_width=True)