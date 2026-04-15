# ✈️ Predictive Maintenance for Turbofan Engines using LSTM

<!-- ## 📌 Overview
This project focuses on predicting the **Remaining Useful Life (RUL)** of turbofan engines using **deep learning techniques**. Accurate RUL prediction enables proactive maintenance, reduces downtime, and improves operational safety in aerospace systems.

The model is trained on the **NASA C-MAPSS dataset** and deployed through an **interactive Streamlit dashboard** for real-time predictions and decision support. -->

## 📌 Overview
This project aims to predict the **Remaining Useful Life (RUL)** of turbofan engines using deep learning techniques. The goal is to estimate how many cycles an engine can operate before failure, enabling timely maintenance decisions.

The model is trained on the **NASA C-MAPSS dataset** and deployed through an interactive **Streamlit dashboard** for real-time prediction and monitoring.

---

##  Key Features

- RUL Prediction using LSTM (Long Short-Term Memory)
- Time-series modeling of engine degradation
- Feature selection using variance analysis and SHAP insights
- Deep learning-based approach outperforming traditional ML models
- Interactive dashboard built with Streamlit
- Engine health classification:
  - Early Life
  - Mid Life
  - Near Failure
- Maintenance recommendations for decision support

---

##  Methodology

### 1. Data Preprocessing
- Computed RUL from engine cycle data
- Normalized sensor values
- Generated sequences (window size = 30) for LSTM input
- Removed low-variance sensors to reduce noise

### 2. Models Implemented
- Linear Regression
- Random Forest
- Support Vector Regression (SVR)
- XGBoost
- **LSTM (Final Model)**

### 3. LSTM Architecture
- Input: Sequence of 30 cycles
- Layers:
  - LSTM (128 units)
  - Dropout (0.3)
  - LSTM (64 units)
  - Dense output layer
- Loss Function: Mean Squared Error

---

## 📊 Results

| Model               | RMSE | MAE |
|--------------------|------|-----|
| Linear Regression  | ~20 |~16|
| Random Forest      | ~16  |~12|
| XGBoost            | ~16  | ~11|
| **LSTM**           | **~13** | **~9** |


<!-- 👉 LSTM achieved the best performance due to its ability to capture **temporal dependencies**. -->
The LSTM model performed best, mainly because it captures temporal dependencies in sequential sensor data.

---

## 🖥️ Dashboard Features

<!-- - Upload engine sensor data (CSV)
- Predict Remaining Useful Life (RUL)
- Display:
  - Engine Life Stage
  - Maintenance Recommendation
- Visual health indicator
- Supports real-time inference -->

The Streamlit dashboard allows users to:
- Upload engine sensor data (CSV)
- Predict RUL for a given engine
- View:
  - Remaining Useful Life
  - Engine life stage
  - Suggested maintenance action

---

## ⚙️ Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- Pandas, NumPy
- Matplotlib

---

## 📡 Deployment

The project is deployed using **Streamlit Community Cloud**, enabling access via a web interface.

👉 *(Add your deployed link here)*

---

## 🎯 Applications

- Aerospace engine monitoring
- Predictive maintenance systems
- Industrial IoT analytics
- Failure prediction in mechanical systems


## ⭐ Acknowledgment
This work was carried out as part of a B.Tech final year project in Mechanical Engineering, focusing on applying machine learning and deep learning techniques to real-world engineering problems.
<!-- This project was developed as part of a B.Tech final year project in Mechanical Engineering, focusing on the intersection of **AI and predictive maintenance**. -->

