import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta

# Load the model and saved predictions
model_path = 'xgb_model.pkl'
predictions_path = 'xgb_predictions.pkl'

st.title("Sales Forecasting and Inventory Management")

try:
    xgb_model = joblib.load(model_path)
    xgb_predictions = joblib.load(predictions_path)
except Exception as e:
    st.error(f"Error loading models or predictions: {e}")
    xgb_predictions = []

st.sidebar.title("Options")
view_section = st.sidebar.radio(
    "Select a section:",
    ["Data Overview", "Predictions Visualization", "Safety Stock Analysis"]
)

if view_section == "Data Overview":
    st.header("Data Overview")
    
    uploaded_file = st.file_uploader("Upload your CSV file with input data", type=["csv"])
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Here is an overview of your data:")
        st.dataframe(data.head())

        st.write("Descriptive statistics:")
        st.write(data.describe())

elif view_section == "Predictions Visualization":
    st.header("Predictions Visualization")
    
    if len(xgb_predictions) == 0:
        st.error("Predictions are unavailable. Ensure the `xgb_predictions.pkl` file is valid.")
    else:
        # User input to display forecasts
        start_date = st.date_input("Start date for forecasts:", datetime.now())
        periods = st.number_input("Number of days to forecast:", min_value=1, max_value=365, value=60)
        
        # Generate future dates
        future_dates = pd.date_range(start=start_date, periods=periods, freq='D')
        
        # Adjust predictions to the requested length
        if len(xgb_predictions) < periods:
            st.warning("The number of predictions is less than the number of requested days.")
            adjusted_predictions = xgb_predictions
            future_dates = future_dates[:len(xgb_predictions)]
        else:
            adjusted_predictions = xgb_predictions[:periods]

        # Visualization of predictions
        st.write("Sales forecasts:")
        plt.figure(figsize=(12, 6))
        plt.plot(future_dates, adjusted_predictions, label="Forecasts", color="blue")
        plt.title("Sales Forecasts for the Next Days")
        plt.xlabel("Date")
        plt.ylabel("Units Sold")
        plt.legend()
        plt.grid()
        st.pyplot(plt)

elif view_section == "Safety Stock Analysis":
    st.header("Safety Stock Analysis")
    
    if len(xgb_predictions) == 0:
        st.error("Predictions are unavailable. Ensure the `xgb_predictions.pkl` file is valid.")
    else:
        lead_time_days = st.number_input("Lead time (days):", min_value=1, max_value=30, value=3)
        service_level = st.slider("Service level (%):", min_value=90, max_value=99, value=95) / 100.0
        
        # Calculate safety stock
        demand_std = np.std(xgb_predictions[:lead_time_days])
        z_score = {0.9: 1.28, 0.95: 1.65, 0.99: 2.33}[service_level]
        safety_stock = z_score * demand_std
        
        st.write(f"Recommended safety stock: {safety_stock:.2f} units.")
        
        # Adjust predictions to match dates
        future_dates = pd.date_range(start=datetime.now(), periods=len(xgb_predictions), freq='D')
        adjusted_predictions = xgb_predictions[:len(future_dates)]

        # Display predictions and safety stock
        plt.figure(figsize=(12, 6))
        plt.axhline(y=safety_stock, color='red', linestyle="--", label="Safety Stock")
        plt.plot(future_dates, adjusted_predictions, label="Forecasted Demand", color="blue")
        plt.title("Forecasted Demand vs Safety Stock")
        plt.xlabel("Date")
        plt.ylabel("Units")
        plt.legend()
        plt.grid()
        st.pyplot(plt)

# Add a section to download the model
st.sidebar.write("Download saved objects:")
try:
    st.sidebar.download_button("Download XGBoost Model", data=open(model_path, "rb"), file_name="xgb_model.pkl")
    st.sidebar.download_button("Download Predictions", data=open(predictions_path, "rb"), file_name="xgb_predictions.pkl")
except Exception as e:
    st.sidebar.error(f"Error preparing downloads: {e}")
