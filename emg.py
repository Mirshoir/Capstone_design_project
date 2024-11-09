import streamlit as st
import pandas as pd
import neurokit2 as nk
from sklearn.preprocessing import StandardScaler
import os
import datetime
import logging

# Suppress warnings
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Directory to store EMG data and analysis results
DATA_STORAGE_DIR = 'uploaded_data_emg'
EMG_RESULTS_FILE = 'emgAnalysisSteps.txt'

# Create directories if they don't exist
os.makedirs(DATA_STORAGE_DIR, exist_ok=True)

# Define sampling rate
emg_sampling_rate = 1000  # in Hz


# Helper function to get the most recent file in the directory
def get_most_recent_file(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    if not files:
        return None
    return max(files, key=os.path.getctime)


# Main function to handle file upload and analysis
def analyze_emg_data():
    st.title("EMG Data Analysis for Muscle Activation Estimation")
    st.write("Upload your EMG (Electromyography) data to analyze muscle activation levels.")

    # File Upload Section
    uploaded_file = st.file_uploader("Choose a CSV file containing your EMG data.", type="csv")

    # Use last saved file if no new file is uploaded
    if uploaded_file is None:
        most_recent_file = get_most_recent_file(DATA_STORAGE_DIR)
        if most_recent_file:
            st.info(f"No new file uploaded. Using the last saved file: `{most_recent_file}`")
            uploaded_file = open(most_recent_file, 'rb')
        else:
            st.warning("No file uploaded and no previous file found. Please upload an EMG CSV file.")
            return

    try:
        # Read and process the uploaded or last saved CSV file
        emg_data = pd.read_csv(
            uploaded_file,
            skiprows=[0, 2],
            names=['Timestamp_Unix_CAL', 'EMG_CAL'],
            usecols=[0, 1],
            low_memory=False
        )

        # Data preprocessing steps
        emg_data.dropna(how='all', inplace=True)
        emg_data['Timestamp_Unix_CAL'] = pd.to_numeric(emg_data['Timestamp_Unix_CAL'], errors='coerce')
        emg_data['EMG_CAL'] = pd.to_numeric(emg_data['EMG_CAL'], errors='coerce')
        emg_data.dropna(inplace=True)
        emg_data['Timestamp'] = pd.to_datetime(emg_data['Timestamp_Unix_CAL'], unit='ms')
        emg_data.set_index('Timestamp', inplace=True)

        # Save the processed data to CSV
        filename = f"emg_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = os.path.join(DATA_STORAGE_DIR, filename)
        emg_data.to_csv(file_path, index=False)
        st.success(f"Data saved as `{filename}` in `{DATA_STORAGE_DIR}`")

        # Perform EMG analysis
        run_emg_analysis(emg_data)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")


# Function to perform EMG data analysis
def run_emg_analysis(emg_data):
    # Initialize the scaler
    scaler = StandardScaler()

    # Standardize the EMG signal
    emg_data['EMG_CAL'] = scaler.fit_transform(emg_data[['EMG_CAL']])

    # Display data samples
    st.subheader('Data Sample')
    st.write(emg_data.head())

    # Plot the EMG signal
    st.subheader('EMG Signal')
    st.line_chart(emg_data['EMG_CAL'])

    # EMG Signal Processing and Muscle Activity Analysis
    st.header('EMG Signal Processing and Muscle Activity Analysis')

    # Clean the EMG signal
    emg_cleaned = nk.emg_clean(emg_data['EMG_CAL'], sampling_rate=emg_sampling_rate)

    # Compute the envelope of the EMG signal (using RMS)
    emg_envelope = nk.emg_amplitude(emg_cleaned)
    emg_data['EMG_Envelope'] = emg_envelope

    # Plot the EMG envelope
    st.subheader('EMG Envelope (Muscle Activity Level)')
    st.line_chart(emg_data['EMG_Envelope'])

    # Estimate Muscle Activation Level
    st.header('Muscle Activation Level Estimation')

    # Define thresholds for muscle activation
    activation_threshold = emg_data['EMG_Envelope'].mean() + emg_data['EMG_Envelope'].std()

    # Identify periods of high muscle activation
    emg_data['High_Activation'] = emg_data['EMG_Envelope'] > activation_threshold

    # Visualize muscle activation over time
    st.subheader('Estimated Muscle Activation Over Time')
    st.line_chart(emg_data['High_Activation'].astype(int))

    # Display percentage of time under high muscle activation
    high_activation_percentage = emg_data['High_Activation'].mean() * 100
    st.write(f"Percentage of time under high muscle activation: {high_activation_percentage:.2f}%")

    # Explanation of Muscle Activation Levels
    st.subheader('Explanation of Muscle Activation Levels')
    st.write("""
    - The **EMG Envelope** represents the overall muscle activity level by taking the root mean square (RMS) of the EMG signal.
    - The **activation threshold** is calculated as the mean plus one standard deviation of the envelope values. This threshold helps identify periods of significant muscle activity.
    - **High Activation** indicates that the muscle is exerting force above the normal resting level, suggesting periods of physical effort or tension.
    """)

    # Save analysis results to a text file
    with open(EMG_RESULTS_FILE, 'w') as results_file:
        results_file.write("EMG Analysis Results\n")
        results_file.write(f"Data Sample:\n{emg_data.head().to_string()}\n\n")
        results_file.write("Percentage of time under high muscle activation: "
                           f"{high_activation_percentage:.2f}%\n")
        results_file.write("Explanation of Muscle Activation Levels:\n")
        results_file.write("""
        - The EMG Envelope represents the overall muscle activity level.
        - The activation threshold helps identify periods of significant muscle activity.
        - High Activation suggests periods of physical effort or tension.
        """)

