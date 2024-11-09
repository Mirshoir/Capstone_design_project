import streamlit as st
import pandas as pd
import neurokit2 as nk
import os
import datetime
import matplotlib.pyplot as plt

# Directory for saving ECG data and analysis steps
DATA_STORAGE_DIR = 'uploaded_data_ecg'
ECG_RESULTS_FILE = 'ecgAnalysisSteps.txt'

# Create directories if they don't exist
os.makedirs(DATA_STORAGE_DIR, exist_ok=True)


# Helper function to get the most recent file in the directory
def get_most_recent_file(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    if not files:
        return None
    return max(files, key=os.path.getctime)


# Main function for analyzing ECG data
def analyze_ecg_data():
    st.title("ECG Data Analysis for Stress Level Estimation")
    st.write("Upload your ECG (Electrocardiogram) data in CSV format to analyze stress levels.")

    # File upload section
    uploaded_file = st.file_uploader("Choose a CSV file containing your ECG data.", type="csv")

    if uploaded_file is None:
        # Use the last saved file if no new file is uploaded
        most_recent_file = get_most_recent_file(DATA_STORAGE_DIR)
        if most_recent_file:
            st.info(f"No new file uploaded. Using the last saved file: `{most_recent_file}`")
            uploaded_file = open(most_recent_file, 'rb')
        else:
            st.warning("No file uploaded and no previous file found. Please upload an ECG CSV file.")
            return

    try:
        # Load ECG data from the uploaded or last saved file
        ecg_data = pd.read_csv(
            uploaded_file,
            names=['Timestamp_Unix_CAL', 'ECG_CAL'],
            usecols=[0, 1],
            low_memory=False
        )

        # Convert 'Timestamp_Unix_CAL' to numeric, setting non-numeric values to NaN
        ecg_data['Timestamp_Unix_CAL'] = pd.to_numeric(ecg_data['Timestamp_Unix_CAL'], errors='coerce')
        ecg_data['ECG_CAL'] = pd.to_numeric(ecg_data['ECG_CAL'], errors='coerce')

        # Remove rows with NaN values
        ecg_data.dropna(inplace=True)

        # Convert 'Timestamp_Unix_CAL' to datetime format (in milliseconds) and set as index
        ecg_data['Timestamp'] = pd.to_datetime(ecg_data['Timestamp_Unix_CAL'], unit='ms', errors='coerce')
        ecg_data.dropna(subset=['Timestamp'], inplace=True)
        ecg_data.set_index('Timestamp', inplace=True)

        # Save cleaned data to a CSV file in DATA_STORAGE_DIR
        filename = f"ecg_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = os.path.join(DATA_STORAGE_DIR, filename)
        ecg_data.to_csv(file_path, index=False)
        st.success(f"Data saved as `{filename}` in `{DATA_STORAGE_DIR}`")

        # Initialize results file
        with open(ECG_RESULTS_FILE, 'w') as results_file:
            results_file.write("ECG Data Loaded Successfully.\n")
            results_file.write(f"Data Sample:\n{ecg_data.head().to_string()}\n\n")

        # Standardize the ECG signal for analysis
        ecg_data['ECG_CAL'] = (ecg_data['ECG_CAL'] - ecg_data['ECG_CAL'].mean()) / ecg_data['ECG_CAL'].std()
        st.line_chart(ecg_data['ECG_CAL'])

        # Record standardized signal to results file
        with open(ECG_RESULTS_FILE, 'a') as results_file:
            results_file.write("Standardized ECG Signal:\n")
            results_file.write(ecg_data['ECG_CAL'].head().to_string() + "\n\n")

        # Perform ECG signal cleaning and R-peak detection
        ecg_cleaned = nk.ecg_clean(ecg_data['ECG_CAL'], sampling_rate=1000)
        ecg_peaks, _ = nk.ecg_peaks(ecg_cleaned, sampling_rate=1000)
        r_peak_indices = ecg_peaks['ECG_R_Peaks']
        st.write(f"Number of R-peaks detected: {len(r_peak_indices)}")

        with open(ECG_RESULTS_FILE, 'a') as results_file:
            results_file.write(f"Number of R-peaks detected: {len(r_peak_indices)}\n")

        # Plot ECG with R-peaks using a figure object
        if len(r_peak_indices) > 0:
            fig, ax = plt.subplots()
            time_ecg = ecg_data.index
            ax.plot(time_ecg, ecg_cleaned, label='ECG Signal')
            ax.scatter(time_ecg[r_peak_indices], ecg_cleaned[r_peak_indices], color='red', label='R-peaks')
            ax.set_xlabel('Time')
            ax.set_ylabel('ECG Amplitude')
            ax.legend()
            st.pyplot(fig)

        # Compute Heart Rate Variability (HRV) features
        try:
            ecg_hrv = nk.hrv(ecg_peaks, sampling_rate=1000, show=False)
            st.subheader("HRV Analysis")
            st.write(ecg_hrv)

            with open(ECG_RESULTS_FILE, 'a') as results_file:
                results_file.write("HRV Analysis:\n")
                results_file.write(ecg_hrv.to_string() + "\n")

            # Estimate Stress Level from HRV (using SDNN as an example)
            sdnn = ecg_hrv['HRV_SDNN'].values[0] if 'HRV_SDNN' in ecg_hrv.columns else None
            stress_level = "Unknown"
            if sdnn is not None:
                # Define stress levels based on SDNN thresholds
                sdnn_mean, sdnn_std = ecg_hrv['HRV_SDNN'].mean(), ecg_hrv['HRV_SDNN'].std()
                if sdnn < (sdnn_mean - sdnn_std):
                    stress_level = "High Stress"
                elif sdnn > (sdnn_mean + sdnn_std):
                    stress_level = "Low Stress"
                else:
                    stress_level = "Moderate Stress"

            st.subheader("Estimated Stress Level")
            st.write(f"Estimated Stress Level: **{stress_level}**")

            with open(ECG_RESULTS_FILE, 'a') as results_file:
                results_file.write(f"Estimated Stress Level: {stress_level}\n")

        except Exception as e:
            st.error("An error occurred during HRV calculation.")
            with open(ECG_RESULTS_FILE, 'a') as results_file:
                results_file.write(f"Error during HRV calculation: {e}\n")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        with open(ECG_RESULTS_FILE, 'a') as results_file:
            results_file.write(f"Error: {e}\n")
