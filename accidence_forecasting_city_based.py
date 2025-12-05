#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, to_date, count, avg, dayofweek, month, when, col, lag, weekofyear, year, lit, quarter, log1p
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.sql import functions as F


# In[2]:


# configure spark session
spark = SparkSession.builder.appName("AccidentForecasting").getOrCreate()
spark.sparkContext.setLogLevel("FATAL")

# load data
df = spark.read.csv("/storage/home/mbr5956/work/Project/sampled_accident_data.csv", header=True, inferSchema=True)


# In[3]:


# df.printSchema()


# In[4]:


df = df.dropna()


# In[5]:


# feature selection
cols = ["ID", "Start_Time", "Severity", "City", "State", "Temperature(F)", "Humidity(%)", 
        "Visibility(mi)", "Precipitation(in)", "Weather_Condition"]
df = df.select(*cols)


# In[6]:


# date-time feature handling
df = df.withColumn("Start_Time", to_timestamp("Start_Time"))
df = df.withColumn("Date", to_date("Start_Time"))

# --- City-level accident hotspots ---
top_cities = (
    df.groupBy("City")
      .agg(F.count("ID").alias("Total_Accidents"))
      .orderBy(F.desc("Total_Accidents"))
      .limit(20)
)

top_cities_pd = top_cities.toPandas()

plt.figure(figsize=(10, 6))
plt.barh(top_cities_pd["City"], top_cities_pd["Total_Accidents"])
plt.title("Top 20 Cities With the Most Accidents")
plt.xlabel("Accident Count")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("top_cities_hotspots.png")

# In[7]:


# df.show(5)


# In[8]:


daily_counts = (
    df.groupBy("City", "Date")
      .agg(count("ID").alias("Accident_Count"))
)

weather_agg = (
    df.groupBy("City", "Date")
      .agg(
          avg("Temperature(F)").alias("Temp"),
          avg("Humidity(%)").alias("Humidity"),
          avg("Visibility(mi)").alias("Visibility"),
          avg("Precipitation(in)").alias("Precip"),
          avg("Severity").alias("Sever"),
      )
)

data = daily_counts.join(weather_agg, ["City", "Date"], "left")

from pyspark.ml.clustering import KMeans

# --- City-level clustering by weather + severity ---
city_features = (
    df.groupBy("City")
      .agg(
          F.avg("Severity").alias("avg_severity"),
          F.avg("Temperature(F)").alias("avg_temp"),
          F.avg("Humidity(%)").alias("avg_humidity"),
          F.avg("Visibility(mi)").alias("avg_visibility"),
          F.avg("Precipitation(in)").alias("avg_precip")
      )
)

assembler_city = VectorAssembler(
    inputCols=["avg_severity", "avg_temp", "avg_humidity", "avg_visibility", "avg_precip"],
    outputCol="features"
)

city_features = assembler_city.transform(city_features)

kmeans = KMeans(k=5, seed=42)
kmeans_model = kmeans.fit(city_features)

clustered = kmeans_model.transform(city_features)
cluster_pd = clustered.toPandas()

plt.figure(figsize=(10, 6))
plt.scatter(
    cluster_pd["avg_temp"], 
    cluster_pd["avg_humidity"],
    c=cluster_pd["prediction"],
    cmap="viridis"
)
plt.xlabel("Average Temperature (F)")
plt.ylabel("Average Humidity (%)")
plt.title("City Clusters Based on Weather & Severity Patterns")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.savefig("city_cluster_weather_severity.png")

# In[9]:


# data.show(5)


# In[10]:


# # Collect data from Spark to driver
# rows = data.select("Date", "Accident_Count").orderBy("Date").collect()

# # Convert to numpy arrays
# dates = np.array([r["Date"] for r in rows])
# accidents = np.array([r["Accident_Count"] for r in rows], dtype=float)

# # Plot
# plt.figure(figsize=(10, 5))
# plt.plot(dates, accidents, marker='o')
# plt.title("Accident Count Over Time")
# plt.xlabel("Date")
# plt.ylabel("Accident Count")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


# In[11]:


data = data.withColumn("day_of_week", dayofweek("Date"))
data = data.withColumn("month", month("Date"))
data = data.withColumn("is_weekend", when(col("day_of_week").isin([1,7]), 1).otherwise(0))
data = data.withColumn("quarter", quarter("Date"))


# In[12]:


# year_selected = 2022
# week_selected = 10

# # Filter data for that week
# week_data = data.filter(
#     (year("Date") == lit(year_selected)) &
#     (weekofyear("Date") == lit(week_selected))
# ).orderBy("Date")

# # Collect for plotting
# rows = week_data.select("Date", "Accident_Count").collect()
# dates = np.array([r["Date"] for r in rows])
# counts = np.array([r["Accident_Count"] for r in rows], dtype=float)

# # Plot
# plt.figure(figsize=(10, 5))
# plt.plot(dates, counts, marker='o', color='green')
# plt.title(f"Accident Count for Week {week_selected} of {year_selected}")
# plt.xlabel("Date")
# plt.ylabel("Accident Count")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


# In[13]:


windowSpec = Window.partitionBy("City").orderBy("Date")

# create lag features of accident_count for time-series forecasting
data = data.withColumn("lag_1", lag("Accident_Count", 1).over(windowSpec))
data = data.withColumn("lag_7", lag("Accident_Count", 7).over(windowSpec))
data = data.na.drop(subset=["lag_1", "lag_7"])
data = data.withColumn("Accident_Count_Log", log1p("Accident_Count")) # using log-transform to prevent negative value prediction


# In[14]:


# data.show(5)


# In[15]:


# data splitting
train  = data.filter(col("Date") < "2021-01-01")
valid  = data.filter((col("Date") >= "2021-01-01") & (col("Date") < "2022-01-01"))
test   = data.filter(col("Date") >= "2022-01-01")


# In[16]:


# Collect to driver for all three splits
train_rows = (
    train.select("Date", "Accident_Count")
         .orderBy("Date")
         .collect()
)

valid_rows = (
    valid.select("Date", "Accident_Count")
         .orderBy("Date")
         .collect()
)

test_rows = (
    test.select("Date", "Accident_Count")
        .orderBy("Date")
        .collect()
)

# Convert to numpy arrays
train_dates = np.array([r["Date"] for r in train_rows], dtype='datetime64[D]')
train_counts = np.array([r["Accident_Count"] for r in train_rows], dtype=float)

valid_dates = np.array([r["Date"] for r in valid_rows], dtype='datetime64[D]')
valid_counts = np.array([r["Accident_Count"] for r in valid_rows], dtype=float)

test_dates = np.array([r["Date"] for r in test_rows], dtype='datetime64[D]')
test_counts = np.array([r["Accident_Count"] for r in test_rows], dtype=float)

# Plot
plt.figure(figsize=(14, 6))

plt.plot(train_dates, train_counts, label='Train Data', marker='o', linewidth=2)
plt.plot(valid_dates, valid_counts, label='Validation Data', marker='s', linewidth=2)
plt.plot(test_dates, test_counts, label='Test Data', marker='x', linewidth=2)

plt.title("Accident Count Over Time — Train / Validation / Test Split", fontsize=16)
plt.xlabel("Date", fontsize=13)
plt.ylabel("Accident Count", fontsize=13)
plt.legend(fontsize=12)

# Optional: split vertical markers
first_valid = np.min(valid_dates)
first_test = np.min(test_dates)

plt.axvline(first_valid, color='gray', linestyle='--', linewidth=1.5)
plt.axvline(first_test, color='gray', linestyle='--', linewidth=1.5)

plt.text(first_valid, max(train_counts)*0.95, " Start Validation", rotation=90, color='gray')
plt.text(first_test, max(train_counts)*0.95, " Start Test", rotation=90, color='gray')

plt.grid(True, linestyle='--', alpha=0.4)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("forecast_split_plot.png") 
# plt.show()


# In[17]:


feature_cols = ["day_of_week", "month", "is_weekend", 
                "Temp", "Humidity", "Visibility", "Precip", "Sever",
                "lag_1", "lag_7", "quarter"]


# In[18]:



# full pipeline

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

gbt = GBTRegressor(
    featuresCol="features",
    labelCol="Accident_Count_Log",
    seed=42
)

pipeline = Pipeline(stages=[assembler, gbt])

# Example hyperparameter options
log_messages = []


# hyperparameter fine-tuning
maxDepths = [3, 4, 5, 6]
maxIters = [80, 100, 120, 150]
stepSizes = [0.01, 0.03, 0.05, 0.07, 0.1]

best_rmse = float("inf")
best_model = None
best_params = None

for depth in maxDepths:
    for it in maxIters:
        for step in stepSizes:
            msg = f"Training with maxDepth={depth}, maxIter={it}, stepSize={step}"
            log_messages.append(msg)

            # Update GBT parameters
            gbt.setParams(maxDepth=depth, maxIter=it, stepSize=step)

            # Train
            model = pipeline.fit(train)

            # Evaluate on validation (log scale only)
            predictions = model.transform(valid)

            rmse = RegressionEvaluator(
                labelCol="Accident_Count_Log",
                predictionCol="prediction",
                metricName="rmse"
            ).evaluate(predictions)

            msg = f"→ RMSE: {rmse}"
            log_messages.append(msg)

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_params = (depth, it, step)

summary = (
    f"Best RMSE: {best_rmse}\n"
    f"Best Params: maxDepth={best_params[0]}, "
    f"maxIter={best_params[1]}, stepSize={best_params[2]}"
)

# print(summary)
log_messages.append(summary)

# Save best model
best_model.write().overwrite().save("cluster_models/best_gbt_model")





# In[19]:


# # get feature importances
# gbt_stage = best_model.stages[1]

# fi = gbt_stage.featureImportances
# features = feature_cols
# pd.DataFrame({"Feature": features, "Importance": fi.toArray()}).sort_values(by="Importance", ascending=False)

# --- Feature importance visualization ---
gbt_stage = best_model.stages[1]
fi = gbt_stage.featureImportances.toArray()

plt.figure(figsize=(10, 6))
plt.barh(feature_cols, fi)
plt.title("Feature Importance — Accident Forecasting Model")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("forecast_feature_importance.png")

# In[20]:


final_preds = best_model.transform(test)

# Reverse log only AFTER model selection
final_preds = final_preds.withColumn(
    "predicted_count",
    F.expm1("prediction")   # undo log1p
)


# Evaluate on LOG SCALE

evaluator_log = RegressionEvaluator(
    labelCol="Accident_Count_Log",
    predictionCol="prediction",
    metricName="rmse"
)

test_rmse = evaluator_log.evaluate(final_preds)

print(f"LOG RMSE on testing data: {test_rmse}")
log_messages.append(f"LOG RMSE on testing data: {test_rmse}")


# Write log to cluster output

sc = spark.sparkContext
sc.parallelize(log_messages, 1).saveAsTextFile("cluster_output/manual_hyperparam_log1")


# Add rounded prediction

final_preds = final_preds.withColumn(
    "predicted_count_rounded",
    F.when(F.round("predicted_count") < 0, 0)
     .otherwise(F.round("predicted_count"))
     .cast("integer")
)


# Convert to Pandas

pdf = final_preds.toPandas()


# FIX: Ensure Date column is 1-D

def flatten(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return x[0]
    return x

pdf["Date"] = pdf["Date"].apply(flatten)


# Extract columns

dates = pd.to_datetime(pdf["Date"]).to_numpy()
actual = pdf["Accident_Count"].astype(float).to_numpy()
predicted = pdf["predicted_count_rounded"].astype(float).to_numpy()


# Sort by date

sort_idx = np.argsort(dates)
dates = dates[sort_idx]
actual = actual[sort_idx]
predicted = predicted[sort_idx]


# Plot

plt.figure(figsize=(16, 8))
plt.plot(dates, actual, label='Actual', marker='o', markersize=5)
plt.plot(dates, predicted, label='Predicted', linestyle='--', marker='x', markersize=5)
plt.xlabel('Date')
plt.ylabel('Accident Count')
plt.title('Accident Occurrence Forecast')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("cluster_forecast_test.png")

# plt.show()

# --- City-specific forecast plots for top N cities ---
# Reuse the top_cities frame if you still have it in scope.
# If not, recompute quickly here:
top_cities_local = (
    pdf.groupby("City")["Accident_Count"]
       .sum()
       .sort_values(ascending=False)
       .head(5)
       .index
       .tolist()
)

for city in top_cities_local:
    pdf_city = pdf[pdf["City"] == city].copy()
    pdf_city = pdf_city.sort_values("Date")

    for city in top_cities_local:
        pdf_city = pdf[pdf["City"] == city].copy()
    if pdf_city.empty:
        continue  # no test data for this city, skip

    pdf_city = pdf_city.sort_values("Date")

    # Convert to numpy arrays to avoid pandas 2.x + matplotlib indexing issue
    dates_city = pd.to_datetime(pdf_city["Date"]).to_numpy()
    actual_city = pdf_city["Accident_Count"].to_numpy(dtype=float)
    predicted_city = pdf_city["predicted_count_rounded"].to_numpy(dtype=float)

    plt.figure(figsize=(14, 6))
    plt.plot(dates_city, actual_city, label="Actual", marker="o")
    plt.plot(dates_city, predicted_city, label="Predicted", linestyle="--", marker="x")
    plt.title(f"Accident Forecast — {city}")
    plt.xlabel("Date")
    plt.ylabel("Accident Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"cluster_forecast_{city.replace('/', '_')}.png")
    plt.close()

    plt.title(f"Accident Forecast — {city}")
    plt.xlabel("Date")
    plt.ylabel("Accident Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"cluster_forecast_{city.replace('/', '_')}.png")

# In[21]:


spark.stop()


# In[ ]:




