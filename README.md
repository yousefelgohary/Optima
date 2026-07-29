<div align="center">
  <h1>⚡ Project Optima - Cluster Intelligence</h1>
  
  ![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=flat&logo=python)
  ![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg?style=flat&logo=streamlit)
  ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)
  ![License](https://img.shields.io/badge/License-MIT-green.svg)
</div>

<br>

👉 **[Live Streamlit App](INSERT_STREAMLIT_APP_LINK_HERE)**

## 📖 Project Overview
Project Optima is an interactive, production-ready MLOps dashboard designed for advanced cluster intelligence. It provides deep visibility into cluster resource metrics, temporal capacity trends, and features an integrated Machine Learning prediction engine to proactively forecast CPU and RAM requests based on granular job specifications.

## ✨ Key Features
- **📊 Interactive EDA (Exploratory Data Analysis):** Explore multi-dimensional cluster trace data (Borg traces) with dynamic filtering. Visualize correlation matrices, temporal load distributions (e.g., Peak Hour Load Maps), and feature distributions using interactive Plotly and Seaborn charts.
- **⚙️ Prediction Engine:** A state-of-the-art inference pipeline powered by **HistGradientBoostingRegressor** models. By specifying job parameters (Scheduling Class, Priority, Event Type, etc.), the engine predicts the exact CPU and RAM resource requests, displaying capacity utilisation limits dynamically using responsive Gauge and Metric UI components.

## 🛠️ Tech Stack
- **Frontend & Routing:** Streamlit (Custom Dark Mode UI & Custom Navigation)
- **Data Processing:** Pandas, NumPy
- **Visualisation:** Plotly Graph Objects, Seaborn, Matplotlib
- **Machine Learning:** Scikit-Learn (HistGradientBoosting), Joblib

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone git@github.com:yousefelgohary/Optima.git
   cd Optima
   ```

2. **Install the dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app locally:**
   ```bash
   streamlit run app.py
   ```

## 📁 Repository Structure
```
Optima/
├── app.py                        # Main landing page for the Streamlit dashboard
├── pages/
│   ├── 2_📊_Interactive_EDA.py    # Exploratory Data Analysis & visualisations
│   └── 3_⚙️_Prediction_Engine.py # ML inference & capacity utilisation metrics
├── src/
│   ├── data_loader.py            # Cached data parsing & temporal feature extraction
│   ├── model_inference.py        # Model loading & HGBR prediction logic
│   ├── preprocessing.py          # Data scaling & feature engineering for inference
│   ├── ui_components.py          # Custom CSS, layout components, and Plotly charts
│   └── config.py                 # Centralised constants and file paths
├── optima_models/                # Serialised ML models (joblib) & scaler
├── datasets/                     
│   └── borg_traces_data.csv      # Sample trace dataset
├── notebooks/
│   └── Project_Optima_Cluster_Resource_Prediction.ipynb  # Original Jupyter notebook
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## 📄 License
This project is licensed under the MIT License.
