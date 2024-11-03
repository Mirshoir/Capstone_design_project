import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from sklearn.preprocessing import StandardScaler
import logging
import os


# ------------------------------------------
# Dashboard Main Function
# ------------------------------------------
def run_dashboard():
    # Suppress warnings
    global emotional_state
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    # ------------------------------------------
    # Setup and Configurations
    # ------------------------------------------

    # Define sampling rates
    gsr_sampling_rate = 1000  # GSR sampling rate in Hz
    ppg_sampling_rate = 1000  # PPG sampling rate in Hz

    # Initialize Standard Scaler
    scaler = StandardScaler()

    # Directory for storing uploaded data
    DATA_STORAGE_DIR = 'uploaded_data'
    os.makedirs(DATA_STORAGE_DIR, exist_ok=True)

    # ------------------------------------------
    # Streamlit Interface
    # ------------------------------------------
    st.title("GSR and PPG Data Analysis for Cognitive Load and Emotional State")
    st.write("Upload your GSR and PPG data in CSV format to analyze cognitive load and emotional state.")

    # ------------------------------------------
    # File Upload Section
    # ------------------------------------------
    st.header('Upload Your Data')
    # Single file uploader for both GSR and PPG data
    uploaded_file = st.file_uploader("Choose a CSV file containing GSR and PPG data", type="csv", key="gsr_file")

    if uploaded_file is not None:
        try:
            # Read and preprocess CSV data
            gsr_ppg_data = pd.read_csv(
                uploaded_file,
                skiprows=[0, 2],
                names=['Timestamp_Unix_CAL', 'GSR_Skin_Conductance_CAL', 'PPG_A13_CAL', 'PPGtoHR_CAL'],
                usecols=[0, 1, 2, 3],
                low_memory=False
            )
            gsr_ppg_data.dropna(how='all', inplace=True)
            for col in ['Timestamp_Unix_CAL', 'GSR_Skin_Conductance_CAL', 'PPG_A13_CAL', 'PPGtoHR_CAL']:
                gsr_ppg_data[col] = pd.to_numeric(gsr_ppg_data[col], errors='coerce')
            gsr_ppg_data.dropna(inplace=True)
            gsr_ppg_data['Timestamp'] = pd.to_datetime(gsr_ppg_data['Timestamp_Unix_CAL'], unit='ms')
            gsr_ppg_data.set_index('Timestamp', inplace=True)

            # Standardize GSR and PPG signals
            gsr_ppg_data['GSR_Skin_Conductance_CAL'] = scaler.fit_transform(gsr_ppg_data[['GSR_Skin_Conductance_CAL']])
            gsr_ppg_data['PPG_A13_CAL'] = scaler.fit_transform(gsr_ppg_data[['PPG_A13_CAL']])

            # ------------------------------------------
            # Data Display and Visualization
            # ------------------------------------------
            st.header('Data Analysis Results')
            st.subheader('Data Sample')
            st.write(gsr_ppg_data.head())
            st.subheader('GSR Signal')
            st.line_chart(gsr_ppg_data['GSR_Skin_Conductance_CAL'])
            st.subheader('PPG Signal')
            st.line_chart(gsr_ppg_data['PPG_A13_CAL'])

            # ------------------------------------------
            # GSR Signal Processing for Cognitive Load
            # ------------------------------------------
            st.header('GSR Signal Processing and Cognitive Load Estimation')
            eda_signals, _ = nk.eda_process(gsr_ppg_data['GSR_Skin_Conductance_CAL'], sampling_rate=gsr_sampling_rate)
            gsr_ppg_data['EDA_Tonic'] = eda_signals['EDA_Tonic']
            gsr_ppg_data['EDA_Phasic'] = eda_signals['EDA_Phasic']
            st.subheader('EDA Tonic Component')
            st.line_chart(gsr_ppg_data['EDA_Tonic'])
            st.subheader('EDA Phasic Component')
            st.line_chart(gsr_ppg_data['EDA_Phasic'])

            # Cognitive load estimation
            scl_mean = gsr_ppg_data['EDA_Tonic'].mean()
            scl_std = gsr_ppg_data['EDA_Tonic'].std()
            gsr_ppg_data['High_Cognitive_Load'] = gsr_ppg_data['EDA_Tonic'] > (scl_mean + scl_std)
            high_load_percentage = gsr_ppg_data['High_Cognitive_Load'].mean() * 100
            st.subheader('Estimated Cognitive Load Over Time')
            st.line_chart(gsr_ppg_data['High_Cognitive_Load'].astype(int))
            st.write(f"Percentage of time under high cognitive load: {high_load_percentage:.2f}%")

            # ------------------------------------------
            # PPG Signal Processing for Emotional State
            # ------------------------------------------
            st.header('PPG Signal Processing and Emotional State Estimation')
            ppg_cleaned = nk.ppg_clean(gsr_ppg_data['PPG_A13_CAL'], sampling_rate=ppg_sampling_rate)
            ppg_peaks = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=ppg_sampling_rate)
            num_peaks = len(ppg_peaks['PPG_Peaks'])
            st.write(f"Number of PPG peaks detected: {num_peaks}")

            if num_peaks > 0:
                # Plot PPG with peaks
                fig, ax = plt.subplots()
                time_ppg = np.arange(len(ppg_cleaned)) / ppg_sampling_rate
                ax.plot(time_ppg, ppg_cleaned, label='PPG Signal')
                ax.scatter(time_ppg[ppg_peaks['PPG_Peaks']], ppg_cleaned[ppg_peaks['PPG_Peaks']], color='red',
                           label='Peaks')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.legend()
                st.pyplot(fig)

                # Compute HRV from PPG
                try:
                    ppg_hrv = nk.hrv(ppg_peaks, sampling_rate=ppg_sampling_rate, show=False)
                    st.subheader('PPG HRV Features')
                    st.write(ppg_hrv)

                    # Emotional state estimation based on HRV
                    sdnn = ppg_hrv['HRV_SDNN'].values[0]
                    sdnn_mean, sdnn_std = ppg_hrv['HRV_SDNN'].mean(), ppg_hrv['HRV_SDNN'].std()
                    sdnn_thresholds = (sdnn_mean - sdnn_std, sdnn_mean + sdnn_std)
                    if sdnn < sdnn_thresholds[0]:
                        emotional_state = 'Sadness or Stress'
                    elif sdnn > sdnn_thresholds[1]:
                        emotional_state = 'Happiness or Relaxation'
                    else:
                        emotional_state = 'Neutral'
                    st.subheader('Estimated Emotional State')
                    st.write(f"Estimated emotional state: **{emotional_state}**")
                except Exception as e:
                    st.error(f"Error calculating HRV from PPG: {e}")
            else:
                st.warning("No peaks detected in PPG signal for HRV computation.")

            # Save results
            with open("dashboard_results.txt", "w") as f:
                f.write(f"Emotional State: {emotional_state}\nHigh Cognitive Load: {high_load_percentage:.2f}%")

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info('Please upload a CSV file to proceed.')


# Run dashboard
run_dashboard()
