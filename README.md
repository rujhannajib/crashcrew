# 🚗 Car Accident Analysis — crashcrew

**DS/CMPSC 410 — Fall 2025**

## 📌 Overview

This project analyzes large-scale U.S. road accident data using **PySpark** to identify spatial and temporal accident patterns, build predictive models, and evaluate performance on a multi-node cluster.

Road accidents generate millions of records annually. Traditional tools struggle with the dataset size (7M+ rows), so distributed computing is essential for scalable preprocessing, analysis, and modeling.

## ⚙️ Tech Stack

- **PySpark** (main framework)
- **MLlib** for clustering & ML models
- **Matplotlib** for visualization
- **Cluster setup:** 1–4 nodes (for scaling experiments)
- **Streamlit:** Dashboard

## 📂 Dataset

**Source:** Kaggle — US Accidents (2016–2023)  
**Records:** 7.73 million  
**Size:** 3.06 GB  
**Working subset:** 35,000 cleaned rows (for local development)

Features include:

- **Geospatial:** latitude, longitude, state
- **Temporal:** hour, weekday, season
- **Environmental:** precipitation, visibility, weather conditions

## 🎯 Project Objectives

- Preprocess large-scale accident data using PySpark
- Engineer temporal, spatial, and environmental features
- Run scalable clustering (e.g., k-means) to find accident hotspots
- Perform forecasting/classification models (severity & accident counts)
- Generate visualizations (heatmaps, maps, cluster plots)

## 👥 Team (Crash Crew)

- **Rujhan Najib** — Data acquisition, preprocessing, forecasting model
- **Joshua George** — ML pipeline, severity prediction
- **Minh Khoi Duong** — ML pipeline
- **Jonathan Park** — Visualization & results
- **Aqil Harith Ramsaid** — Documentation & cluster experimentation

**Workstreams:**

- **ML Pipeline:** Rujhan, Joshua, Minh
- **Cluster Workflow:** Aqil, Jonathan

## ⚠️ Challenges & Mitigation

- **Large dataset size** → use sampling during development
- **Missing data** → define clear cleaning rules

# 🖥️ How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/rujhannajib/crashcrew.git
```

## 2. Activate the Environment

Activate the .venv environment to run the webpage.

### Windows (Command Prompt)

```bash
.venv\Scripts\activate.bat
```

### Windows (PowerShell)

```
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```
source .venv/bin/activate
```

To run the Python files for each machine learning model, activate your Jupyter Notebook or your cluster’s Python environment.

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run Python Files on the Cluster

Enter the cluster:

```bash
ssh your_id@psu.edu
```

Navigate to the project directory:

```bash
cd /path/to/project
```

Change the python file in standalone.sh. Configure appropriate information (CPU, nodes, ntask email)
Submit the job:

```bash
sbatch standalone.sh
```

## 5. Running the frontend

Run streamlit:

```bash
streamlit run app.py
```

## 6. Generate sampled data

Upload the full dataset into project directory

```bash
#!/bin/bash
curl -L -o ~/Downloads/us-accidents.zip\
  https://www.kaggle.com/api/v1/datasets/download/sobhanmoosavi/us-accidents
```

To perform sampling on the huge dataset: run Sampling.ipynb

# Webpage

app.py: Streamlit entry file

- pages/city_analytics.py: Accidence Analysis based on city
- pages/city_cluster.py: Accidence Analysis based on city
- pages/daily_forecasting.py: Daily forecasting analysis
- pages/hourly_forecasting.py: Hourly forecasting analysis
- pages/severity.py: Severity analysis

# Daily Accidence Forecasting

Jupyter Notebook: accidence_forecasting.ipynb
Python file: accidence_forecasting.py

Generated output:

- cluster_models: pyspark model
- cluster_output: hyperparameter tuning
- cluster_test: test dataset exported
- clusterdailyforecast_test.png: Prediction result in PNG format
- forecast_feature_importance.png: Output feature importance from cluster training

# City-Level Accidence Forecasting

Python file: accidence_forecasting_city_based.py

Generated output:

- cluster_models: pyspark model
- cluster_output: hyperparameter tuning
- cluster_test: test dataset exported
- clusterdailyforecast_test.png: Prediction result in PNG format
- forecast_feature_importance.png: Output feature importance from cluster training
- city_cluster_weather_severity.png: Output cluster hotspot from K-Means Clustering
- cluster_forecast_Houston.png: Prediction result in PNG for a sample city (Houston)
- forecast_feature_importance.png: Output feature importance from cluster training but for the whole sample
- top_cities_hotspots.png: Distribution of Accidents based on the US Cities

## 📚 References

US Accidents Dataset:

- https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

Streamlit Documentation

- https://docs.streamlit.io/

Time Series Forecasting with XGBoost - Use python and machine learning to predict energy consumption

- https://youtu.be/vV12dGe_Fho?si=-nzIc4qDapFDBkbL

Project Presentation

- https://psu.mediaspace.kaltura.com/media/My+Meeting/1_wii2c1vk​
