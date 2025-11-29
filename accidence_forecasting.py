#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, to_date, count, avg, dayofweek, month, when, col, lag, weekofyear, year, lit, quarter
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline


# In[2]:



spark = SparkSession.builder.appName("AccidentForecasting").getOrCreate()
spark.sparkContext.setLogLevel("FATAL")

df = spark.read.csv("cleaned_accident_data.csv", header=True, inferSchema=True)


# In[3]:


# df.printSchema()


# In[4]:

df = df.dropna()


cols = ["ID", "Start_Time", "Severity", "City", "State", "Temperature(F)", "Humidity(%)", 
        "Visibility(mi)", "Precipitation(in)", "Weather_Condition"]
df = df.select(*cols)


# In[5]:


df = df.withColumn("Start_Time", to_timestamp("Start_Time"))
df = df.withColumn("Date", to_date("Start_Time"))


# In[6]:


# df.show(5)


# In[7]:


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


# In[8]:


# data.show(5)


# In[9]:


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


# In[10]:


data = data.withColumn("day_of_week", dayofweek("Date"))
data = data.withColumn("month", month("Date"))
data = data.withColumn("is_weekend", when(col("day_of_week").isin([1,7]), 1).otherwise(0))
data = data.withColumn("quarter", quarter("Date"))


# In[11]:


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


# In[12]:


windowSpec = Window.partitionBy("City").orderBy("Date")

data = data.withColumn("lag_1", lag("Accident_Count", 1).over(windowSpec))
data = data.withColumn("lag_7", lag("Accident_Count", 7).over(windowSpec))
data = data.na.drop(subset=["lag_1", "lag_7"])


# In[13]:


# data.show(5)


# In[14]:


train  = data.filter(col("Date") < "2021-01-01")
valid  = data.filter((col("Date") >= "2021-01-01") & (col("Date") < "2022-01-01"))
test   = data.filter(col("Date") >= "2022-01-01")

# export test dataset for testing
test.write.mode("overwrite").option("header", True).csv("cluster_test.csv")



# In[15]:


# # Collect to driver for all three splits
# train_rows = (
#     train.select("Date", "Accident_Count")
#          .orderBy("Date")
#          .collect()
# )

# valid_rows = (
#     valid.select("Date", "Accident_Count")
#          .orderBy("Date")
#          .collect()
# )

# test_rows = (
#     test.select("Date", "Accident_Count")
#         .orderBy("Date")
#         .collect()
# )

# # Convert to numpy arrays
# train_dates = np.array([r["Date"] for r in train_rows], dtype='datetime64[D]')
# train_counts = np.array([r["Accident_Count"] for r in train_rows], dtype=float)

# valid_dates = np.array([r["Date"] for r in valid_rows], dtype='datetime64[D]')
# valid_counts = np.array([r["Accident_Count"] for r in valid_rows], dtype=float)

# test_dates = np.array([r["Date"] for r in test_rows], dtype='datetime64[D]')
# test_counts = np.array([r["Accident_Count"] for r in test_rows], dtype=float)

# # Plot
# plt.figure(figsize=(14, 6))

# plt.plot(train_dates, train_counts, label='Train Data', marker='o', linewidth=2)
# plt.plot(valid_dates, valid_counts, label='Validation Data', marker='s', linewidth=2)
# plt.plot(test_dates, test_counts, label='Test Data', marker='x', linewidth=2)

# plt.title("Accident Count Over Time — Train / Validation / Test Split", fontsize=16)
# plt.xlabel("Date", fontsize=13)
# plt.ylabel("Accident Count", fontsize=13)
# plt.legend(fontsize=12)

# # Optional: split vertical markers
# first_valid = np.min(valid_dates)
# first_test = np.min(test_dates)

# plt.axvline(first_valid, color='gray', linestyle='--', linewidth=1.5)
# plt.axvline(first_test, color='gray', linestyle='--', linewidth=1.5)

# plt.text(first_valid, max(train_counts)*0.95, " Start Validation", rotation=90, color='gray')
# plt.text(first_test, max(train_counts)*0.95, " Start Test", rotation=90, color='gray')

# plt.grid(True, linestyle='--', alpha=0.4)
# plt.xticks(rotation=45)
# plt.tight_layout()

# plt.show()


# In[16]:

feature_cols = ["day_of_week", "month", "is_weekend", 
                "Temp", "Humidity", "Visibility", "Precip", "Sever",
                "lag_1", "lag_7", "quarter"]


# In[17]:



# full pipeline

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

gbt = GBTRegressor(
    featuresCol="features",
    labelCol="Accident_Count",
    seed=42
)

pipeline = Pipeline(stages=[assembler, gbt])


evaluator_rmse = RegressionEvaluator(
    labelCol="Accident_Count",
    predictionCol="prediction",
    metricName="rmse"
)


# Example hyperparameter options
log_messages = []

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
            print(msg); log_messages.append(msg)

            # Update GBT parameters
            gbt.setParams(maxDepth=depth, maxIter=it, stepSize=step)

            # Train
            model = pipeline.fit(train)

            # Validate
            predictions = model.transform(valid)
            rmse = evaluator_rmse.evaluate(predictions)

            msg = f"→ RMSE: {rmse}"
            print(msg); log_messages.append(msg)

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

# Write log to cluster output
sc = spark.sparkContext
sc.parallelize(log_messages, 1).saveAsTextFile("cluster_output/manual_hyperparam_log")

# Save best model
best_model.write().overwrite().save("cluster_models/best_gbt_model")





# In[18]:



# In[19]:


# gbt_stage = best_model.stages[1]

# fi = gbt_stage.featureImportances
# features = feature_cols
# pd.DataFrame({"Feature": features, "Importance": fi.toArray()}).sort_values(by="Importance", ascending=False)


# # In[20]:


# pdf = predictions.toPandas().to_numpy()

# # Extract columns by position
# dates = pdf[:, 1]                 # column 1 = Date
# actual = pdf[:, 2].astype(float)  # column 2 = Accident_Count (convert to float)
# predicted = pdf[:, -1].astype(float)  # last column = prediction

# # Optional: sort by date to ensure the line plot is continuous
# sort_idx = np.argsort(dates)
# dates = np.array(dates)[sort_idx]
# actual = actual[sort_idx]
# predicted = predicted[sort_idx]

# # Plot
# plt.figure(figsize=(10, 5))
# plt.plot(dates, actual, label='Actual', marker='o')
# plt.plot(dates, predicted, label='Predicted', linestyle='--', marker='x')
# plt.xlabel('Date')
# plt.ylabel('Accident Count')
# plt.title('Accident Occurrence Forecast')
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


# In[21]:


spark.stop()

