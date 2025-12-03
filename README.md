# 🚗 Car Accident Analysis — crashcrew

**DS/CMPSC 410 — Fall 2025**

## 📌 Overview

This project analyzes large-scale U.S. road accident data using **PySpark** to identify spatial and temporal accident patterns, build predictive models, and evaluate performance on a multi-node cluster.

Road accidents generate millions of records annually. Traditional tools struggle with the dataset size (7M+ rows), so distributed computing is essential for scalable preprocessing, analysis, and modeling.

## ⚙️ Tech Stack

- **PySpark** (main framework)
- **MLlib** for clustering & ML models
- **Matplotlib/Seaborn** for visualization
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
- Compare runtime across different cluster sizes
- Generate visualizations (heatmaps, maps, cluster plots)

## 🛠️ Pipeline

1. **Data Ingestion**

   - Load CSV → Spark DataFrames
   - Handle missing or invalid fields

2. **Feature Engineering**

   - Temporal (hour, season), geospatial, weather features
   - Partitioning strategies by state or time

3. **Exploratory Analysis**

   - Accident counts by state/year/hour
   - Early density maps & distributions

4. **Modeling**

   - K-means clustering (hotspot detection)
   - Severity prediction
   - Daily accident forecasting

5. **Scaling Experiments**

   - Run models on 1, 2, and 4 nodes
   - Measure runtime & memory usage

6. **Visualization**
   - Export results for plotting
   - Produce heatmaps, geospatial cluster maps

## 📈 Expected Deliverables

- Clean & documented PySpark pipeline
- Scalable ML models (clustering, forecasting, classification)
- Visualizations & hotspot maps
- Final report + presentation slides

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
- **Visualization overhead** → focus on accuracy first
- **Cluster limits** → optimize partitioning & caching strategy

## 📚 References

US Accidents Dataset:

- https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
