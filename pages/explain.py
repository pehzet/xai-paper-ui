import streamlit as st
import json
import pandas as pd
import os
from .data_table import show_data_table
def show_explanation():
    # Labels und Testdaten aus dem Session State

    last_key = next(reversed(st.session_state.choices))
    selected_label = st.session_state.choices[last_key]
    true_label = st.session_state.true_label
    test_case = st.session_state.test_case  # sollte ein dict oder DataFrame-kompatibel sein
    # Optional: Lob, wenn korrekt
    if selected_label == true_label:
        st.success("Contragulations! Your choice is correct! 🎉")
        st.session_state.current_choice_is_correct = True
    else:
        st.session_state.current_choice_is_correct = False
        st.error("Your choice is incorrect. Here is an Explanation")
    # Zeige Antwortvergleich
    st.markdown(f"**Your answer:** {selected_label}")
    st.markdown(f"**Correct answer:** {true_label}")
    st.markdown("__"*20)
    # Zeige Testdaten als Tabelle
    st.markdown("**Case Features:**")
    df = pd.DataFrame(test_case.items(), columns=["Feature", "Value"])
    df = df.set_index("Feature")
    # st.table(df)
    st.dataframe(df, column_config={
        "widgets": st.column_config.Column(
            width="medium"
        )
    })
    st.markdown("__"*20)
    # zeige Gesamtdaten tabelle
    show_data_table()
    st.markdown("__"*20)
    # Lade Erklärung aus JSON-Datei

    current_dir = os.path.dirname(os.path.abspath(__file__))
    crop_path = os.path.join(current_dir, "crop_explanations.json")

    with open(crop_path, "r", encoding="utf-8") as f:
        crop_info = json.load(f)

    # Erklärung ausgeben
    explanation = crop_info.get(true_label, "No explanation found for this crop.")
    
    st.markdown("**Explanation:**")
    st.write(explanation)


    st.write("If you have read the explanation, please click the button below to continue.")
    if st.button("Continue"):
        st.session_state.decision_completed = True
        # st.session_state["page"] = "chat"
        st.rerun()


