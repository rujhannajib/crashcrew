import streamlit as st

st.write("# Crashcrew Project")

vis_page = st.Page(
    "pages/vis.py", title="Visualization")
ml_models_page = st.Page("pages/accidence_forecast.py", title="Accidence Forecasting")