import streamlit as st
import pandas as pd

st.title("City Climate–Severity Clusters")

st.image("city_cluster_weather_severity.png", caption="City Clusters by Weather & Severity")

st.write("### Cluster Summary Table")
summary_df = pd.read_csv("cluster_weather_severity_summary.csv")
st.dataframe(summary_df)

st.write("---")
st.write("## Cluster Interpretations")

cluster_descriptions = {
    0: "Cold, very humid cities with winter-related severity risks (snow, ice, fog).",
    1: "Extremely cold, saturated cities with high severity due to freezing conditions.",
    2: "Warm, humid southeastern cities with rainfall-driven accident patterns.",
    3: "Mild-climate cities with moderate humidity; weather has limited impact.",
    4: "Hot, dry western cities where severity is driven more by behavior than weather."
}

for cluster_id in sorted(cluster_descriptions.keys()):
    with st.expander(f"Cluster {cluster_id} Details"):
        st.markdown(f"**Description:** {cluster_descriptions[cluster_id]}")

        # Display subtable for this cluster only
        st.write("**Cluster Statistics:**")
        st.dataframe(summary_df[summary_df["Cluster"] == cluster_id])
