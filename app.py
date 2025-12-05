import streamlit as st

st.set_page_config(page_title="Crashcrew Project", layout="wide")

# -------------------------------
# Title + Team Members
# -------------------------------
st.title("🚦 Crashcrew Project")

st.page_link("pages/daily_forecasting.py", label="Daily Forecasting", icon="📊")
st.page_link("pages/hourly_forecasting.py", label="Hourly Forecasting", icon="🚨")

st.markdown("""
### 👥 Team Members  
- **Muhammad Aqil Ramsaid**  
- **Joshua George**  
- **Jonathan Park**  
- **Minh Khoi Duong**  
- **Muhammad Rujhan Najib Bin Fauzi Najib**
""")

# -------------------------------
# Navigation Pages
# -------------------------------
hour_forecast = st.Page("pages/hourly_forecasting.py", title="Hourly Forecasting")
daily_forecast = st.Page("pages/daily_forecasting.py", title="Daily Forecasting")

# -------------------------------
# Objective Section
# -------------------------------
st.subheader("🎯 Objective")

st.markdown("""
Road accidents are among the **leading causes of injuries and fatalities** in the United States, placing significant burdens on public safety, healthcare, and transportation systems.  
With **millions of crashes reported every year**, understanding the patterns and contributing factors behind these accidents is essential for:

- Smarter **road design**
- Better **urban planning**
- Improved **traffic management policies**

Traditional analytical tools struggle because the dataset is:

- **Massive** (millions of rows)
- **Complex** (structured + semi-structured)
- **Multifactor** (temporal, spatial, environmental, infrastructure)

Using **Big Data & Machine Learning**, we can capture these complex patterns and build scalable, data-driven insights into traffic safety.
""")

# -------------------------------
# Data Source Section
# -------------------------------
st.subheader("📚 Data Source")

st.markdown("""
The dataset used in this project is available from Kaggle:

🔗 **US Accidents (3.0 million records)**  
https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
""")
