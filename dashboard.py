# integrated_gsr_ppg_app.py

import os
import time
import csv
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from sklearn.preprocessing import StandardScaler
import streamlit as st
import logging
from serial import Serial, SerialException
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
from serial.tools import list_ports
from queue import Queue

# Suppress warnings from matplotlib
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# ------------------------------------------
# Global Variables and Setup
# ------------------------------------------

# Define sampling rates (adjust as needed)
gsr_sampling_rate = 1000  # in Hz
ppg_sampling_rate = 1000  # in Hz

# Directory to store uploaded or collected data
DATA_STORAGE_DIR = 'uploaded_data_gsr_ppg'

# Create the directory if it doesn't exist
if not os.path.exists(DATA_STORAGE_DIR):
    os.makedirs(DATA_STORAGE_DIR)

# A global queue for inter-thread communication (Shimmer data)
data_queue = Queue()


# ------------------------------------------
# Functions
# ------------------------------------------

def save_emotional_state(emotional_state, output_file='C:\\Users\dtfygu876\\prompt_codes\\csvChunking\\Chatbot_for_Biosensor\\emotional_state.txt'):
    """
    Save the estimated emotional state to a single file, overwriting on each run.
    """
    # Create the directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the current timestamp
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Overwrite the emotional state record in the file
    with open(output_file, 'w') as file:
        file.write(f"Timestamp: {timestamp_str}\n")
        file.write(f"Emotional State: {emotional_state}\n")

    st.success(f"Emotional state has been sent to Chatbot and saved in `{output_file}`.")


def analyze_gsr_ppg_data(gsr_ppg_data):
    """
    Perform analysis on GSR and PPG data, including emotional state estimation.
    """
    scaler = StandardScaler()

    # Standardize the GSR and PPG signals
    gsr_ppg_data['GSR_Skin_Conductance_CAL'] = scaler.fit_transform(gsr_ppg_data[['GSR_Skin_Conductance_CAL']])
    gsr_ppg_data['PPG_A13_CAL'] = scaler.fit_transform(gsr_ppg_data[['PPG_A13_CAL']])

    # Display Data Sample
    st.subheader('Data Sample')
    st.write(gsr_ppg_data.head())

    # Plot the GSR and PPG signals
    st.subheader('GSR Signal')
    st.line_chart(gsr_ppg_data['GSR_Skin_Conductance_CAL'])

    st.subheader('PPG Signal')
    st.line_chart(gsr_ppg_data['PPG_A13_CAL'])

    # ------------------------------------------
    # PPG Signal Processing and Emotional State Estimation
    # ------------------------------------------

    st.header('PPG Signal Processing and Emotional State Estimation')
    ppg_signal = gsr_ppg_data['PPG_A13_CAL'].values
    ppg_cleaned = nk.ppg_clean(ppg_signal, sampling_rate=ppg_sampling_rate)
    ppg_peaks = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=ppg_sampling_rate)
    ppg_peak_indices = ppg_peaks['PPG_Peaks']

    if len(ppg_peak_indices) > 0:
        fig, ax = plt.subplots()
        time_ppg = np.arange(len(ppg_cleaned)) / ppg_sampling_rate
        ax.plot(time_ppg, ppg_cleaned, label='PPG Signal')
        ax.scatter(time_ppg[ppg_peak_indices], ppg_cleaned[ppg_peak_indices], color='red', label='Peaks')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        st.pyplot(fig)

        # Compute intervals between peaks (in ms)
        if len(ppg_peak_indices) > 1:
            intervals = np.diff(ppg_peak_indices) * (1000 / ppg_sampling_rate)  # Convert to ms

            if len(intervals) < 2:
                st.write("Not enough intervals for HRV analysis. Please provide more data.")
            else:
                # Compute SDNN (standard deviation of intervals)
                sdnn = np.std(intervals)
                st.subheader('Estimated Time-Domain HRV Feature: SDNN')
                st.write(f"SDNN: {sdnn:.2f} ms")

                # Emotional State Estimation based on SDNN
                # We can define arbitrary thresholds or compare to a baseline.
                # Here, we assume a hypothetical mean and std for demonstration:
                sdnn_mean = np.mean(intervals)  # This is just a placeholder
                sdnn_std = np.std(intervals)  # Also a placeholder

                if sdnn < sdnn_mean - sdnn_std:
                    emotional_state = 'Sadness or Stress'
                elif sdnn > sdnn_mean + sdnn_std:
                    emotional_state = 'Happiness or Relaxation'
                else:
                    emotional_state = 'Neutral'

                st.subheader('Estimated Emotional State')
                st.write(f"Emotional State: **{emotional_state}**")
                save_emotional_state(emotional_state)
        else:
            st.write("Not enough PPG peaks to compute HRV intervals. Please provide more data.")
    else:
        st.write("No PPG peaks detected. Emotional state estimation not possible.")


def list_available_ports():
    """List and return all available COM ports."""
    ports = list_ports.comports()
    return [port.device for port in ports]


def handler(pkt: DataPacket, csv_writer, data_queue) -> None:
    """
    Callback function to handle incoming data packets from the Shimmer device.
    """
    try:
        timestamp = pkt.timestamp_unix

        # Safely extract channel data
        def safe_get(channel_type):
            try:
                return pkt[channel_type]
            except KeyError:
                return None

        cur_value_adc = safe_get(EChannelType.INTERNAL_ADC_13)
        cur_value_accel_x = safe_get(EChannelType.ACCEL_LSM303DLHC_X)
        cur_value_accel_y = safe_get(EChannelType.ACCEL_LSM303DLHC_Y)
        cur_value_accel_z = safe_get(EChannelType.ACCEL_LSM303DLHC_Z)
        cur_value_gsr = safe_get(EChannelType.GSR_RAW)
        # PPG uses INTERNAL_ADC_13 as an example; adjust if needed
        cur_value_ppg = safe_get(EChannelType.INTERNAL_ADC_13)
        cur_value_gyro_x = safe_get(EChannelType.GYRO_MPU9150_X)
        cur_value_gyro_y = safe_get(EChannelType.GYRO_MPU9150_Y)
        cur_value_gyro_z = safe_get(EChannelType.GYRO_MPU9150_Z)

        # Write data to the CSV file
        csv_writer.writerow([
            timestamp, cur_value_adc,
            cur_value_accel_x, cur_value_accel_y, cur_value_accel_z,
            cur_value_gsr, cur_value_ppg,
            cur_value_gyro_x, cur_value_gyro_y, cur_value_gyro_z
        ])

        # Put data into the queue for the main thread to read
        data_queue.put((timestamp, cur_value_adc, cur_value_accel_x, cur_value_accel_y, cur_value_accel_z,
                        cur_value_gsr, cur_value_ppg, cur_value_gyro_x, cur_value_gyro_y, cur_value_gyro_z))

    except Exception as e:
        print(f"Unexpected error in handler: {e}")


def run_streaming(username, selected_port, duration_seconds):
    """Run the Shimmer data streaming session and save to a CSV named <username>.csv."""
    csv_file_path = os.path.join(DATA_STORAGE_DIR, f"{username}.csv")
    # If the file exists, remove it before starting a new session
    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)

    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header row
        csv_writer.writerow([
            "Timestamp", "ADC_Value", "Accel_X", "Accel_Y", "Accel_Z",
            "GSR_Value", "PPG_Value", "Gyro_X", "Gyro_Y", "Gyro_Z"
        ])

        try:
            print(f"Connecting to {selected_port}...")
            serial_conn = Serial(selected_port, DEFAULT_BAUDRATE)
            shim_dev = ShimmerBluetooth(serial_conn)

            # Initialize Shimmer device
            shim_dev.initialize()
            dev_name = shim_dev.get_device_name()
            print(f"Connected to Shimmer device: {dev_name}")

            # Add callback for incoming data
            shim_dev.add_stream_callback(lambda pkt: handler(pkt, csv_writer, data_queue))

            # Start streaming
            print("Starting data streaming...")
            shim_dev.start_streaming()
            time.sleep(duration_seconds)
            shim_dev.stop_streaming()
            print("Stopped data streaming.")

            # Shut down the device connection
            shim_dev.shutdown()
            print("Shimmer device connection closed.")
            print("Data collection complete!")

        except SerialException as e:
            print(f"Serial Error: {e}")
        except ValueError as e:
            print(f"Invalid COM port: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    return csv_file_path


def gsr_ppg_app():
    """
    Streamlit app for uploading and analyzing GSR and PPG data.
    """
    st.title("GSR and PPG Data Analysis")

    # ------------------------------------------
    # Data Acquisition from Shimmer Device
    # ------------------------------------------
    st.header("Data Acquisition from Shimmer Device")
    user_name = st.text_input("Enter your name for data labeling:", "")
    available_ports = list_available_ports()
    port_name = st.selectbox("Select the COM port for the Shimmer device:", available_ports)
    stream_duration = st.number_input("Enter the streaming duration in seconds:", min_value=1, value=5)

    if st.button("Start Streaming"):
        if user_name.strip() == "":
            st.warning("Please enter a valid name before starting.")
        elif not port_name:
            st.warning("No COM ports available or selected.")
        else:
            with st.spinner("Collecting data from Shimmer device..."):
                csv_file_path = run_streaming(user_name.strip(), port_name, stream_duration)
            st.success("Data collection completed!")
            st.write(f"Data saved as `{os.path.basename(csv_file_path)}` in `{DATA_STORAGE_DIR}`.")
            # Display some sample data from the queue
            collected_samples = []
            while not data_queue.empty():
                collected_samples.append(data_queue.get())
            if collected_samples:
                st.write("Sample Data Collected:")
                st.write(collected_samples[:5])  # Show first few samples

    # ------------------------------------------
    # Upload or Use Existing Data
    # ------------------------------------------
    st.header("Upload or Use Existing Data for Analysis")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    def process_and_analyze(dataframe):
        # Rename columns if they exist
        column_mapping = {
            'Timestamp': 'Timestamp_Unix_CAL',
            'GSR_Value': 'GSR_Skin_Conductance_CAL',
            'PPG_Value': 'PPG_A13_CAL'
        }
        for orig_col, new_col in column_mapping.items():
            if orig_col in dataframe.columns:
                dataframe.rename(columns={orig_col: new_col}, inplace=True)

        required_columns = ['Timestamp_Unix_CAL', 'GSR_Skin_Conductance_CAL', 'PPG_A13_CAL']

        if all(col in dataframe.columns for col in required_columns):
            # Convert timestamp and set index
            dataframe.dropna(inplace=True)
            dataframe['Timestamp_Unix_CAL'] = pd.to_numeric(dataframe['Timestamp_Unix_CAL'], errors='coerce')
            dataframe['Timestamp'] = pd.to_datetime(dataframe['Timestamp_Unix_CAL'], unit='ms', errors='coerce')
            dataframe.set_index('Timestamp', inplace=True)

            # Save the processed data
            filename = f"gsr_ppg_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            dataframe.to_csv(os.path.join(DATA_STORAGE_DIR, filename))
            st.write(f"Data saved as `{filename}` in `{DATA_STORAGE_DIR}`.")

            # Analyze the processed data
            analyze_gsr_ppg_data(dataframe)
        else:
            st.error("Selected file does not have the required columns.")

    # If a file is uploaded, process it
    if uploaded_file:
        try:
            gsr_ppg_data = pd.read_csv(uploaded_file)
            process_and_analyze(gsr_ppg_data)
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Upload a file to start analysis or select from previously saved data below.")

    # ------------------------------------------
    # Re-analyze Stored Data
    # ------------------------------------------
    st.header("Re-analyze Stored Data")
    stored_files = [f for f in os.listdir(DATA_STORAGE_DIR) if f.endswith('.csv')]

    if stored_files:
        selected_file = st.selectbox("Select a file", stored_files)
        if st.button("Analyze Selected File"):
            try:
                # Read the CSV without index_col so that Timestamp remains a column
                gsr_ppg_data = pd.read_csv(os.path.join(DATA_STORAGE_DIR, selected_file))
                process_and_analyze(gsr_ppg_data)
            except Exception as e:
                st.error(f"Error processing stored file: {e}")
    else:
        st.write("No stored files available.")


if __name__ == "__main__":
    gsr_ppg_app()
