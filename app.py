import streamlit as st
from chatBot import run_chatbot
from dashboard import run_dashboard

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Dashboard", "Chatbot"])

# Display selected page
if page == "Dashboard":
    st.header("Dashboard for GSR and PPG Analysis")
    try:
        run_dashboard()  # Runs the dashboard function
    except Exception as e:
        st.error(f"Error loading dashboard: {e}")
elif page == "Chatbot":
    st.header("Emotional Support Chatbot")
    try:
        run_chatbot()  # Runs the chatbot function
    except Exception as e:
        st.error(f"Error loading chatbot: {e}")
