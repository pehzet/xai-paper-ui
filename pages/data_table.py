import pandas as pd
from pathlib import Path
import streamlit as st


def show_data_table():
    st.markdown("Optimal Values for Parameters")
    st.markdown("This table shows the quartiles (Q1 and Q3) for the parameters used in the prediction model.")
    st.markdown("These values were determined based on the training data and are used to make predictions.")
    FILE_PATH = Path("prediction_model/data/optimal_ranges.csv")

    table = pd.read_csv(FILE_PATH, header=[0, 1], index_col=0)
    table = table.round(2)  
    table.index.name = ""                   
    table.columns.names = ["Feature", ""]    
    UNIT_MAP = {
        "N":          "N (kg/ha)",
        "P":          "P (kg/ha)",
        "K":          "K (kg/ha)",
        "temperature":"temperature (°C)",
        "humidity":   "humidity (%)",
        "ph":         "pH",                   #
        "rainfall":   "rainfall (mm/month)",
    }

    # Multi-Header umbenennen – nur Ebene 0!
    table_disp = table.rename(columns=UNIT_MAP, level=0)
    # 3) In Streamlit darstellen
    st.dataframe(table_disp)                    
    st.markdown(
    """
    <style>
    /* komplette Toolbar entfernen (Suche, Maximieren & Download) */
    [data-testid="stElementToolbar"] {
        display: none !important;
    }

    /* ─ Alternative ─
       Nur den Download-Button verstecken, Suche & Maximieren behalten: 
    div[data-testid="stElementToolbar"] button[title="Download as CSV"] {
        display: none !important;
    }
    */
    </style>
    """,
    unsafe_allow_html=True
)