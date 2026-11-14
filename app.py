import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math

st.set_page_config(page_title="Temperature Forecasting App", layout="wide")

st.title("📈 Temperature Forecasting Dashboard")
st.write("Upload your dataset to visualize historical trends and model predictions.")

# 1. File Uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # Check necessary columns
    if 'date' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['date'])

        # Select temperature column or fallback
        if 'temp' in df.columns:
            target_col = 'temp'
        else:
            st.warning("'temp' column not found. Defaulting to 5th column.")
            target_col = df.columns[4]
            df['temp'] = df[target_col]

        # Sidebar configuration
        st.sidebar.header("Configuration")
        split_ratio = st.sidebar.slider("Train/Validation Split Ratio", 0.5, 0.9, 0.8, 0.05)

        # Split logic
        training_data_len = math.ceil(len(df) * split_ratio)
        val_len = len(df) - training_data_len

        # Dummy model placeholder / predictions loader
        predictions = np.full((val_len, 1), df['temp'].mean())

        # Data processing for plotting
        train_dates = df['date_parsed'].iloc[:training_data_len]
        valid_dates = df['date_parsed'].iloc[training_data_len:]

        valid_plot_with_date = pd.DataFrame(index=valid_dates)
        valid_plot_with_date['temp'] = df['temp'].iloc[training_data_len:].values
        valid_plot_with_date['Predictions'] = predictions

        # Plotting
        st.subheader("Model Performance & Projections")
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(train_dates, df['temp'].iloc[:training_data_len], label='Train')
        ax.plot(valid_plot_with_date.index, valid_plot_with_date['temp'], label='Actual Val')
        ax.plot(valid_plot_with_date.index, valid_plot_with_date['Predictions'], label='Predictions')

        # Formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        fig.autofmt_xdate()
        ax.set_title('Model Temperature Predictions (with Dates)')
        ax.set_xlabel('Date (Month-Year)', fontsize=14)
        ax.set_ylabel('Temperature', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.5)

        st.pyplot(fig)

        # KPI Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Data Points", len(df))
        with col2:
            st.metric("Validation Set Size", val_len)

    else:
        st.error("The uploaded CSV must contain a 'date' column.")
else:
    st.info("Please upload a CSV file to begin.")
