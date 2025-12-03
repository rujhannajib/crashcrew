#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import pyspark
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

from pyspark.sql import SparkSession
from pyspark.sql.functions import hour, to_timestamp, unix_timestamp
import pyspark.sql.functions as F

from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

# Start Spark
ss = SparkSession.builder.appName("accident_duration_prediction").getOrCreate()
ss.sparkContext.setLogLevel("WARN")

# Read the CSV (all columns as strings)
data = ss.read.csv("US_Accident_March23.csv", header=True, inferSchema=False)


# In[2]:


#print("Original schema:")
#data.printSchema()

# Drop rows with any nulls
data = data.dropna()

#print("Sample rows after dropna:")
#data.show(5, truncate=False)

# Select useful columns
df = data.select(
    "ID",
    "Start_Time",
    "End_Time",
    "Severity",
    "State",
    F.col("Distance(mi)").alias("DistanceRaw")
)


# In[3]:


# Convert times to timestamp type
df = df.withColumn("Start_ts", to_timestamp("Start_Time"))
df = df.withColumn("End_ts", to_timestamp("End_Time"))

# Compute duration in minutes
df = df.withColumn(
    "Duration_minutes",
    (unix_timestamp("End_ts") - unix_timestamp("Start_ts")) / 60.0
)

# Extract hour of day from start time
df = df.withColumn("Hour", hour("Start_ts"))

# Cast distance to numeric
df = df.withColumn("Distance", F.col("DistanceRaw").cast("double"))

# Filter out bad or extreme durations (negative or longer than 12 hours)
df = df.filter(
    (F.col("Duration_minutes") > 0) &
    (F.col("Duration_minutes") < 12 * 60)
)

# Keep only the columns we will use for modeling and drop any remaining nulls
model_df = df.select(
    "Hour",
    "Severity",
    "State",
    "Distance",
    "Duration_minutes"
).dropna()

#print("Rows left for modeling:", model_df.count())
#print("Schema for modeling:")
#model_df.printSchema()
#model_df.show(5, truncate=False)


# In[4]:


# Split into train and validation sets
train_df, valid_df = model_df.randomSplit([0.8, 0.2], seed=42)

# Index categorical features
severity_indexer = StringIndexer(
    inputCol="Severity",
    outputCol="Severity_idx",
    handleInvalid="keep"
)

state_indexer = StringIndexer(
    inputCol="State",
    outputCol="State_idx",
    handleInvalid="keep"
)

# Assemble features into a single vector
assembler = VectorAssembler(
    inputCols=["Hour", "Severity_idx", "State_idx", "Distance"],
    outputCol="features"
)

# Define hyperparameter grids
maxDepths = [3, 4, 5, 6]
minInstances = [2, 3]

# Evaluator for RMSE
evaluator = RegressionEvaluator(
    labelCol="Duration_minutes",
    predictionCol="prediction",
    metricName="rmse"
)

best_rmse = float("inf")
best_model = None
best_depth = None
best_minIns = None

# For recording results in a pandas table
hyperparams_eval_df = pd.DataFrame(
    columns=["maxDepth", "minInstancesPerNode", "training_rmse", "valid_rmse"]
)
index = 0


# In[5]:


# Hyperparameter search
for depth in maxDepths:
    for ins in minInstances:
        #print(f"\nTraining model with maxDepth={depth}, minInstancesPerNode={ins}")
        
        # IMPORTANT FIX: set maxBins high enough for State (50 categories)
        dt = DecisionTreeRegressor(
            labelCol="Duration_minutes",
            featuresCol="features",
            maxDepth=depth,
            minInstancesPerNode=ins,
            maxBins=128   # fix for "categorical feature has 50 values" error
        )
        
        pipeline = Pipeline(stages=[severity_indexer, state_indexer, assembler, dt])
        model = pipeline.fit(train_df)
        
        train_pred = model.transform(train_df)
        valid_pred = model.transform(valid_df)
        
        train_rmse = evaluator.evaluate(train_pred)
        valid_rmse = evaluator.evaluate(valid_pred)
        
        #print("Training RMSE:", train_rmse)
        #print("Validation RMSE:", valid_rmse)
        
        hyperparams_eval_df.loc[index] = [depth, ins, train_rmse, valid_rmse]
        index += 1
        
        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            best_model = model
            best_depth = depth
            best_minIns = ins

#print("\nHyperparameter results table:")
#print(hyperparams_eval_df)

#print(
    "\nBest maxDepth =", best_depth,
    ", best minInstancesPerNode =", best_minIns,
    ", best validation RMSE =", best_rmse
)


# In[ ]:


# Use the best model on the validation set
best_predictions = best_model.transform(valid_df)

#print("\nSample predictions on validation set:")
best_predictions.select(
    "Hour",
    "Severity",
    "State",
    "Distance",
    "Duration_minutes",
    "prediction"
).show(20, truncate=False)

# Plot actual vs predicted duration for a sample of points
# Plot actual vs predicted average duration per hour (like your accident count graph)

# 1. Compute average actual and predicted duration per hour
avg_by_hour = (
    best_predictions
    .groupBy("Hour")
    .agg(
        F.avg("Duration_minutes").alias("Actual_Avg_Duration"),
        F.avg("prediction").alias("Predicted_Avg_Duration")
    )
    .orderBy("Hour")
)

avg_by_hour.show()

# 2. Convert to Pandas for plotting
avg_pd = avg_by_hour.toPandas().sort_values("Hour")

hours = avg_pd["Hour"].to_numpy()
actual_avg = avg_pd["Actual_Avg_Duration"].to_numpy()
pred_avg = avg_pd["Predicted_Avg_Duration"].to_numpy()

# 3. Line plot like your severity prediction graph
plt.figure(figsize=(12, 6))

plt.plot(hours, actual_avg, marker="o", label="Actual Avg Duration (min)")
plt.plot(hours, pred_avg, marker="x", linestyle="--", label="Predicted Avg Duration (min)")

plt.title("Actual vs Predicted Average Accident Duration per Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Duration (minutes)")
plt.xticks(hours)  # show each hour on x axis
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("duration_prediction.png", dpi=100)
#plt.show()


# In[ ]:


ss.stop()


# In[ ]:





# In[ ]:





# In[ ]:




