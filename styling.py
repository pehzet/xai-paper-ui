import streamlit as st
def set_page_styling():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
    }

    .stTextInput, .stNumberInput, .stSelectSlider {
        margin-bottom: 20px;
    }

    .stTextInput input, .stNumberInput input {
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 4px;
    }

    .stSelectSlider {
        padding: 10px 0;
    }
    .StyledThumbValue.st-emotion-cache-132r2mp.ew7r33m2 {
        font-family: 'Roboto', sans-serif;
        color: darkgrey;
    }
    .st-emotion-cache-1inwz65.ew7r33m0 {
        font-family: 'Roboto', sans-serif;
        color: darkgrey;
    }

    .st-emotion-cache-1s3l9q9.e1nzilvr5 {
        font-family: 'Roboto', sans-serif;
        color: black;
    }


    .stButton button:hover {
        background-color: #45a049;
    }

    h1 {
        font-weight: 700;
    }

    label {
        font-weight: 400;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)
