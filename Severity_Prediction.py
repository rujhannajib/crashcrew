#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pyspark
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv


# In[3]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType, FloatType
from pyspark.sql.functions import hour, to_timestamp
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, IndexToString
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator, RegressionEvaluator


# In[4]:


ss=SparkSession.builder.appName("severity_prediction").getOrCreate()


# In[17]:


data = ss.read.csv("US_Accidents_March23.csv", header=True, inferSchema=False)


# In[19]:


#data.printSchema()


# In[20]:


data = data.dropna()


# In[21]:


#data.show(5)


# In[22]:


cols = ["ID", "Start_Time", "Severity", "State"]
data = data.select(*cols)


# In[23]:


data = data.withColumn("Hours", hour("Start_Time"))


# In[24]:


#data.show(5)


# In[25]:


data_with_counts = (
    data.groupBy("Hours", "Severity")
    .agg(F.count("*").alias("Accident Count"))
    .orderBy("Hours", "Severity", "Accident Count")
)

#data_with_counts.show(10)


# In[33]:


# Loop through severity level to output plot based on each severity
for i in range(1, 5):
    data_per_severity = data_with_counts.filter(data_with_counts["Severity"] == i).orderBy("Hours").collect()

    # Convert to numpy arrays
    times = np.array([r["Hours"] for r in data_per_severity])
    severities = np.array([r["Accident Count"] for r in data_per_severity], dtype=float)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.scatter(times, severities)
    plt.title(f"Accident Count With Severity {i} Per Hour")
    plt.xlabel("Hours")
    plt.ylabel("Accident Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"cluster_accident_count_with_severity_{i}_per_hour", dpi=300, bbox_inches="tight")
#    plt.show()


# In[27]:


counts_per_hour = (
    data_with_counts
    .groupBy("Hours")
    .agg(F.sum("Accident Count").alias("Accident Count"))
    .orderBy("Hours")
)

#counts_per_hour.show()


# In[28]:


train_df, valid_df= counts_per_hour.randomSplit([0.8, 0.2], seed=42)
hyperparams_eval_df = pd.DataFrame(
    columns=["maxDepth", "minInstancesPerNode", "training_rmse", "valid_rmse", "BestModel"]
)
index = 0
maxDepths = [3, 4, 5, 6]
minInstances = [2, 3]
best_rmse = float("inf")

assembler = VectorAssembler(
    inputCols=["Hours"],
    outputCol="features"
)

evaluator = RegressionEvaluator(
    labelCol="Accident Count",
    predictionCol="prediction",
    metricName="rmse"
)

best_model = None

for depth in maxDepths:
    for ins in minInstances:
        dt = DecisionTreeRegressor(
            labelCol="Accident Count",
            featuresCol="features",
            maxDepth=depth,
            minInstancesPerNode=ins
        )
        pipeline = Pipeline(stages=[assembler, dt])
        model = pipeline.fit(train_df)

        training_predictions = model.transform(train_df)
        valid_predictions = model.transform(valid_df)

        training_rmse = evaluator.evaluate(training_predictions)
        valid_rmse = evaluator.evaluate(valid_predictions)

        hyperparams_eval_df.loc[index] = [depth, ins, training_rmse, valid_rmse, 0]
        index += 1
        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            best_max_depth = depth
            best_min_ins = ins
            best_index = index - 1
            best_model = model

print(
    "Best maxDepth =", best_max_depth,
    ", best minInstancesPerNode =", best_min_ins,
    ", validation RMSE =", best_rmse
)


# In[29]:


trained_df = best_model.transform(counts_per_hour)

trained_df.select("Hours", "Accident Count", "prediction").orderBy("Hours").show()


# In[31]:


result = (
    trained_df
    .select("Hours", "Accident Count", "prediction")
    .orderBy("Hours")
    .toPandas()
)

# Sort just to be safe
result = result.sort_values("Hours")
result.to_csv("cluster_accident_results.csv", index=False)
hours = result["Hours"].to_numpy()
actual = result["Accident Count"].to_numpy()
predicted = result["prediction"].to_numpy()

plt.figure(figsize=(12, 6))

plt.plot(hours, actual, marker="o", label="Actual Accident Count")
plt.plot(hours, predicted, marker="x", linestyle="--", label="Predicted Accident Count")

plt.title("Actual vs Predicted Accident Count per Hour")
plt.xlabel("Hour")
plt.ylabel("Accident Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cluster_accident_plot.png", dpi=300, bbox_inches='tight')
#plt.show()


# In[ ]:


ss.stop()


# In[ ]:




