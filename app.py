import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(layout="wide")

# -----------------------------
# LOAD MODELS
# -----------------------------
# rf_model = pickle.load(open("rf_model.pkl","rb"))
lstm_model = load_model("lstm_model.keras", compile= False)

features = [
"sensor_2","sensor_3","sensor_4",
"sensor_7","sensor_8","sensor_9",
"sensor_11","sensor_12","sensor_13",
"sensor_15","sensor_17"
]
SEQ_LENGTH = 30
# -----------------------------
# CSS (FONT FIX 🔥)
# -----------------------------
st.markdown("""
<style>
[data-testid="stMetricValue"] {font-size: 26px !important;}
[data-testid="stMetricLabel"] {font-size: 14px !important;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HELPERS
# -----------------------------
def get_life_stage(rul):
    if rul > 80:
        return "🟢 Early Life"
    elif rul > 40:
        return "🟡 Mid Life"
    else:
        return "🔴 Near Failure"

def maintenance_action(rul):
    if rul > 80:
        return "Continue Operation"
    elif rul > 40:
        return "Schedule Inspection"
    else:
        return "Perform Maintenance Immediately"

# -----------------------------
# HEADER
# -----------------------------
st.title("✈️ Turbofan Engine Health Monitor")
st.markdown("### Predictive Maintenance Dashboard (ML Powered)")

# -----------------------------
# MODEL SELECT
# -----------------------------
# model_choice = st.selectbox("Select Model", ["Random Forest", "LSTM"])

# -----------------------------
# INPUT
# -----------------------------
st.sidebar.header("Input Engine Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Required for LSTM)")
st.sidebar.markdown("### 🔧 Select Engine to Analyze")

# uploaded_file = st.sidebar.file_uploader("Upload CSV (Required for LSTM)")

sensor_values = {}

if uploaded_file is None:
    st.sidebar.write("Manual Input")

    for sensor in features:
        sensor_values[sensor] = st.sidebar.number_input(sensor, value=0.0)

    input_data = pd.DataFrame([sensor_values])

else:
    input_data = pd.read_csv(uploaded_file)


    if 'engine_id' in input_data.columns:
        engine_ids = input_data['engine_id'].unique()
        selected_engine = st.selectbox("Select Engine", engine_ids)

        input_data = input_data[input_data['engine_id'] == selected_engine]
# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🚀 Predict RUL"):

    # ---------------- RF MODEL ----------------
    # if model_choice == "Random Forest":
    #     pred = rf_model.predict(input_data[features])
    #     rul_pred = int(pred[0])

    # ---------------- LSTM MODEL ----------------
    # else:
        # seq_length = 30

        if uploaded_file is not None:
            # use last 30 cycles
            seq = input_data[features].values[-SEQ_LENGTH:]

            # pad if needed
            if len(seq) < SEQ_LENGTH:
                pad = np.zeros((SEQ_LENGTH - len(seq), len(features)))
                seq = np.vstack([pad, seq])

        else:
            # fallback (demo only)
            seq = np.repeat(input_data[features].values, SEQ_LENGTH, axis=0)

        seq = seq.reshape(1, SEQ_LENGTH, len(features))

        pred = lstm_model.predict(seq)
        rul_pred = max (0, int(pred[0][0]))

    # ---------------- OUTPUT ----------------
    # col1, col2, col3 = st.columns(3)
    # col1.metric("Remaining Useful Life", f"{rul_pred} cycles")

    # stage = get_life_stage(rul_pred)
    # col2.metric("Engine Stage", stage)

    # action = maintenance_action(rul_pred)
    # col3.metric("Recommended Action", action)



        col1, col2, col3 = st.columns(3)
        col1.metric("Remaining Useful Life", f"{rul_pred} cycles")
        col2.metric("Engine Stage", get_life_stage(rul_pred))
        col3.metric("Recommended Action", maintenance_action(rul_pred))


    # ALERTS
        if rul_pred > 80:
            st.success("Engine is healthy.")
        elif rul_pred > 40:
            st.warning("Inspection recommended.")
        else:
            st.error("⚠️ Critical condition! Immediate maintenance required.")

    # PROGRESS
        st.subheader("Engine Health Level")
        # st.progress(min(rul_pred/125, 1.0))
        progress_value = max(0, min(rul_pred/125, 1.0))
        st.progress(progress_value)

# -----------------------------
# VISUALIZATION
# -----------------------------
# st.subheader("Engine Degradation Trend (Demo)")

# cycles = np.arange(1,101)
# actual = np.linspace(120,0,100)
# predicted = actual + np.random.normal(0,8,100)

# fig, ax = plt.subplots()
# ax.plot(cycles, actual, label="Actual RUL")
# ax.plot(cycles, predicted, label="Predicted RUL")

# ax.set_xlabel("Cycle")
# ax.set_ylabel("RUL")
# ax.legend()

# st.pyplot(fig)

# -----------------------------
# SENSOR VISUALIZATION
# -----------------------------
# if uploaded_file is not None:
#     st.subheader("Sensor Trends")
#     st.line_chart(input_data[features])

# -----------------------------
# MODEL INFO
# -----------------------------
st.info(f"Model: LSTM | Dataset: NASA C-MAPSS FD001")


# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt





# st.markdown("""
# <style>
# /* Reduce main metric number size */
# [data-testid="stMetricValue"] {
#     font-size: 28px !important;
# }

# /* Reduce label size */
# [data-testid="stMetricLabel"] {
#     font-size: 14px !important;
# }

# /* Reduce delta text (if any) */
# [data-testid="stMetricDelta"] {
#     font-size: 12px !important;
# }
#             [data-testid="stMetric"] {
#     padding: 10px 0px;
# }
# </style>
# """, unsafe_allow_html=True)
# # -----------------------------
# # Load trained model
# # -----------------------------
# model = pickle.load(open("rf_model.pkl","rb"))

# features = [
# "sensor_2","sensor_3","sensor_4",
# "sensor_7","sensor_8","sensor_9",
# "sensor_11","sensor_12","sensor_13",
# "sensor_15","sensor_17"
# ]

# # -----------------------------
# # Helper Functions (NEW 🔥)
# # -----------------------------
# def get_life_stage(rul):
#     if rul > 80:
#         return "🟢 Early Life"
#     elif rul > 40:
#         return "🟡 Mid Life"
#     else:
#         return "🔴 Near Failure"

# def maintenance_action(rul):
#     if rul > 80:
#         return "Continue Operation"
#     elif rul > 40:
#         return "Schedule Inspection"
#     else:
#         return "Perform Maintenance Immediately"

# # -----------------------------
# # App Header
# # -----------------------------
# st.set_page_config(layout="wide")

# st.title("✈️ Turbofan Engine Health Monitor")
# st.markdown("### Predictive Maintenance Dashboard (ML Powered)")

# # -----------------------------
# # Sidebar Input
# # -----------------------------
# st.sidebar.header("Input Engine Sensor Data")

# uploaded_file = st.sidebar.file_uploader("Upload Sensor CSV")

# sensor_values = {}

# if uploaded_file is None:
#     st.sidebar.write("Or enter sensor values manually")

#     for sensor in features:
#         sensor_values[sensor] = st.sidebar.number_input(sensor, value=0.0)

#     input_data = pd.DataFrame([sensor_values])

# else:
#     input_data = pd.read_csv(uploaded_file)

# # -----------------------------
# # Prediction Section
# # -----------------------------
# if st.button("🚀 Predict RUL"):

#     prediction = model.predict(input_data[features])
#     rul_pred = int(prediction[0])

#     # Layout
#     col1, col2, col3 = st.columns(3)

#     # -----------------------------
#     # RUL Display
#     # -----------------------------
#     col1.metric("Remaining Useful Life", f"{rul_pred} cycles")

#     # -----------------------------
#     # Life Stage
#     # -----------------------------
#     stage = get_life_stage(rul_pred)
#     col2.metric("Engine Stage", stage)

#     # -----------------------------
#     # Maintenance Action
#     # -----------------------------
#     action = maintenance_action(rul_pred)
#     col3.metric("Recommended Action", action)

#     # -----------------------------
#     # Alert System (NEW 🔥)
#     # -----------------------------
#     if rul_pred > 80:
#         st.success("Engine is healthy. No immediate action required.")
#     elif rul_pred > 40:
#         st.warning("Engine in mid-life. Inspection recommended.")
#     else:
#         st.error("⚠️ Critical Condition! Immediate maintenance required.")

#     # -----------------------------
#     # RUL Progress Bar
#     # -----------------------------
#     st.subheader("Engine Health Level")
#     st.progress(min(rul_pred/125, 1.0))

# # -----------------------------
# # Visualization Section
# # -----------------------------
# st.subheader("Engine Degradation Trend (Demo)")

# cycles = np.arange(1,101)
# actual = np.linspace(120,0,100)
# predicted = actual + np.random.normal(0,8,100)

# fig, ax = plt.subplots()
# ax.plot(cycles, actual, label="Actual RUL")
# ax.plot(cycles, predicted, label="Predicted RUL")

# ax.set_xlabel("Cycle")
# ax.set_ylabel("Remaining Useful Life")
# ax.legend()

# st.pyplot(fig)

# # -----------------------------
# # Sensor Visualization (NEW 🔥)
# # -----------------------------
# st.subheader("Sensor Input Overview")

# if uploaded_file is not None:
#     st.line_chart(input_data[features])

# # -----------------------------
# # Model Info
# # -----------------------------
# st.info("Model: Random Forest | Dataset: NASA C-MAPSS FD001")




# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt

# # -----------------------------
# # Load trained model
# # -----------------------------
# model = pickle.load(open("rf_model.pkl","rb"))

# # sensors used in model
# features = [
# "sensor_2","sensor_3","sensor_4",
# "sensor_7","sensor_8","sensor_9",
# "sensor_11","sensor_12","sensor_13",
# "sensor_15","sensor_17"
# ]

# # -----------------------------
# # App Header
# # -----------------------------
# st.title("✈️ Turbofan Engine Health Monitor")
# st.markdown("Machine Learning based Remaining Useful Life Prediction")

# # -----------------------------
# # Sidebar Input
# # -----------------------------
# st.sidebar.header("Input Engine Sensor Data")

# uploaded_file = st.sidebar.file_uploader("Upload Sensor CSV")

# sensor_values = {}

# if uploaded_file is None:

#     st.sidebar.write("Or enter sensor values manually")

#     for sensor in features:
#         sensor_values[sensor] = st.sidebar.number_input(sensor, value=0.0)

#     input_data = pd.DataFrame([sensor_values])

# else:

#     input_data = pd.read_csv(uploaded_file)

# # -----------------------------
# # Prediction Section
# # -----------------------------
# if st.button("Predict RUL"):

#     prediction = model.predict(input_data[features])

#     st.subheader("Predicted Remaining Useful Life")

#     st.metric(label="Remaining Useful Life", value=f"{int(prediction[0])} cycles")

# # -----------------------------
# # Visualization Section
# # -----------------------------
# st.subheader("Example Engine Degradation Trend")

# cycles = np.arange(1,101)

# actual = np.linspace(120,0,100)
# predicted = actual + np.random.normal(0,8,100)

# fig, ax = plt.subplots()

# ax.plot(cycles, actual, label="Actual RUL")
# ax.plot(cycles, predicted, label="Predicted RUL")

# ax.set_xlabel("Cycle")
# ax.set_ylabel("Remaining Useful Life")
# ax.legend()

# st.pyplot(fig)

# # -----------------------------
# # Model Info
# # -----------------------------
# st.info("Model: Random Forest | Dataset: NASA C-MAPSS FD001")