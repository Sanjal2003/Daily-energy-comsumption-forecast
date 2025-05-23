import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta

# Load trained XGBoost model
model = joblib.load('xgb_pjm_model.pkl')

st.title('🔮 30-Day Energy Consumption Forecast (XGBoost)')
st.markdown("Enter the energy consumption (MW) for the past 7 days to generate a forecast.")

# User input for last 7 days' consumption
user_lags = []
for i in range(7, 0, -1):
    val = st.number_input(f'Day -{i}', min_value=0.0, value=35000.0, step=100.0, key=f'lag_{i}')
    user_lags.append(val)

if st.button("Predict Next 30 Days"):
    future_predictions = []
    input_sequence = user_lags.copy()

    for _ in range(30):
        input_array = np.array(input_sequence[-7:]).reshape(1, -1)
        pred = model.predict(input_array)[0]
        future_predictions.append(pred)
        input_sequence.append(pred)

    # Create forecast DataFrame
    last_date = pd.to_datetime('today').normalize()
    future_dates = [last_date + timedelta(days=i+1) for i in range(30)]

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast_MW': future_predictions
    }).set_index('Date')

    # Display Forecast Table
    st.subheader("📅 30-Day Forecast Table")
    st.dataframe(forecast_df.style.format("{:.2f}"))

    # Line Plot
    st.subheader("📊 Forecast Line Plot")
    plt.figure(figsize=(12,6))
    plt.plot(forecast_df.index, forecast_df['Forecast_MW'], marker='o', color='orange')
    plt.title("30-Day Energy Consumption Forecast")
    plt.xlabel("Date")
    plt.ylabel("Energy Consumption (MW)")
    plt.grid(True)
    st.pyplot(plt)
