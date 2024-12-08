# app.py

import streamlit as st
from ecg import ecg_app
from emg import emg_app
from dashboard import gsr_ppg_app
from chatBot import run_chatbot

st.set_page_config(
    page_title="Biosignal Data Analysis App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar Navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", [
    "Home",
    "ECG Analysis",
    "EMG Analysis",
    "GSR/PPG Analysis",
    "Chatbot"
])

if selection == "Home":
    st.title("Welcome to the Cognitive load-Biosignal Data Analysis App")
    st.write("""
    This app allows you to analyze ECG, EMG, GSR, and PPG data for various physiological insights.
    Use the menu on the left to navigate between different analyses and tools, including a chatbot for assistance.
    """)
elif selection == "ECG Analysis":
    ecg_app()
elif selection == "EMG Analysis":
    emg_app()
elif selection == "GSR/PPG Analysis":
    gsr_ppg_app()
elif selection == "Chatbot":
    run_chatbot()
