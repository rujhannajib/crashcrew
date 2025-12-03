import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import numpy as np
import altair as alt

st.title("🚦 ML Models")
st.header("📈 Accident Forecasting Model")

# --------------------------
#  Cached Spark Session
# --------------------------
@st.cache_resource
def get_spark_session():
    return (
        SparkSession.builder
        .appName("Forecasting_Model")
        .getOrCreate()
    )

spark = get_spark_session()
spark.sparkContext.setLogLevel("ERROR")

# --------------------------
# Load Data
# --------------------------
st.subheader("🔍 Loaded Test Data")

try:
    tf = spark.read.csv("cluster_test.csv", header=True, inferSchema=True)
    st.dataframe(tf.limit(10).toPandas())
except Exception as e:
    st.error(f"❌ Failed to load CSV: {e}")
    
st.title("📊 Model Evaluation Results")

st.subheader("Best Model Performance")

best_rmse = 0.46390491116724
best_params = {
    "maxDepth": 4,
    "maxIter": 150,
    "stepSize": 0.1
}
log_rmse = 0.4933598006142605

st.metric(label="Best RMSE", value=f"{best_rmse:.6f}")
st.metric(label="Test LOG RMSE", value=f"{log_rmse:.6f}")

st.subheader("Best Hyperparameters")
st.json(best_params)

st.subheader("Accident Count: Train | Test | Validation Split")
st.image("forecast_split_plot.png")

st.subheader("Accident Occurrence Forecast")
st.image("cluster_forecast_test.png")



