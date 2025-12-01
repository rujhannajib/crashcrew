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

# --------------------------
# Load Model
# --------------------------
st.subheader("🤖 Loading Trained GBT Model")

try:
    loaded_model = PipelineModel.load("cluster_models/best_gbt_model")
    st.success("Model successfully loaded.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")

# --------------------------
# Run Prediction
# --------------------------
st.subheader("📊 Generating Predictions")

predictions = loaded_model.transform(tf)

# Evaluate RMSE
evaluator_rmse = RegressionEvaluator(
    labelCol="Accident_Count",
    predictionCol="prediction",
    metricName="rmse"
)

test_rmse = evaluator_rmse.evaluate(predictions)
st.metric("RMSE", f"{test_rmse:.4f}")

pdf = predictions.select("Date", "Accident_Count", "prediction").toPandas()

st.dataframe(pdf)



