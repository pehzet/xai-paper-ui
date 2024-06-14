import streamlit as st
from PIL import Image

def show_images():
    image1 = Image.open(r"images\image_1.png")
    image2 = Image.open(r"images\image_2.png")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image1, caption="Global FI", use_column_width=True)
        
    with col2:
        st.image(image2, caption="Local Explanation", use_column_width=True)