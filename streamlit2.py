import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Title and Description
st.title("Demand Forecasting and Inventory Management")
st.write("This app predicts future demand and analyzes inventory levels based on provided data.")

# Charger le modèle
model_file = "xgb_model.pkl"
try:
    xgb_model = joblib.load(model_file)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    xgb_model = None

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file and xgb_model:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset", data.head())

    # Preprocessing
    st.write("### Preprocessing")
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')

    # Handling object columns
    object_columns = data.select_dtypes(include=['object']).columns
    for col in object_columns:
        if col in ['Units Sold']:  # Ignore target variable
            continue
        data[col] = pd.factorize(data[col])[0]

    # Splitting data into train and test
    train_size = int(0.8 * len(data))
    train = data.iloc[:train_size]
    test = data.iloc[train_size:]

    train_rf_xgb = train.drop(columns=['Units Sold'])
    test_rf_xgb = test.drop(columns=['Units Sold'])

    # Align train and test columns
    train_rf_xgb, test_rf_xgb = train_rf_xgb.align(test_rf_xgb, join='left', axis=1)
    test_rf_xgb = test_rf_xgb.fillna(0)

    st.write("Processed Training Data", train_rf_xgb.head())
    st.write("Processed Test Data", test_rf_xgb.head())

    # Vérifiez si les colonnes des données de test correspondent à celles du modèle
    model_feature_names = xgb_model.get_booster().feature_names

    # Align test data with model feature names
    if set(test_rf_xgb.columns) != set(model_feature_names):
        missing_features = set(model_feature_names) - set(test_rf_xgb.columns)
        extra_features = set(test_rf_xgb.columns) - set(model_feature_names)

        for feature in missing_features:
            test_rf_xgb[feature] = 0
        test_rf_xgb = test_rf_xgb.drop(columns=list(extra_features))

    test_rf_xgb = test_rf_xgb[model_feature_names]

    # Model Evaluation
    st.write("### Model Evaluation")
    try:
        xgb_predictions = xgb_model.predict(test_rf_xgb)
        xgb_score = mean_absolute_error(test['Units Sold'], xgb_predictions)
        st.write(f"Mean Absolute Error (MAE): {xgb_score:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

    # Forecast Future Demand
    future_periods = st.slider("Select forecast period (days):", min_value=30, max_value=120, value=60)
    future_rf_xgb = test_rf_xgb.iloc[-1:].copy()
    future_predictions = []

    for _ in range(future_periods):
        pred = xgb_model.predict(future_rf_xgb)
        future_predictions.append(pred[0])
        future_rf_xgb.iloc[0, :-1] = future_rf_xgb.iloc[0, 1:]
        future_rf_xgb.iloc[0, -1] = pred

    future_dates = pd.date_range(start=test.index[-1] + pd.Timedelta(days=1), periods=future_periods, freq='D')

    # Plot Forecast
    st.write("### Forecasted Demand")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(test.index, test["Units Sold"], label="Actual (Test Data)", color="red", linewidth=2)
    ax.plot(future_dates, future_predictions, label=f"Forecast ({future_periods} Days)", color="blue", linestyle="--")
    ax.set_title("Demand Forecast", fontsize=16)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Units Sold", fontsize=14)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig)

    # Download Forecast
    forecast_df = pd.DataFrame({"Date": future_dates, "Forecasted Demand": future_predictions})
    csv = forecast_df.to_csv(index=False)
    st.download_button("Download Forecast", csv, "forecast.csv", "text/csv")
