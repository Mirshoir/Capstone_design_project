import os
import streamlit as st
import pandas as pd
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma

# Set your LangSmith API key
os.environ["LANGSMITH_API_KEY"] = "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"

# Path to your Ollama API key
ollama_api_key_path = r"C:\Users\dtfygu876\.ollama\id_ed25519.pub"

# Read and set the Ollama API key from the file
if os.path.exists(ollama_api_key_path):
    with open(ollama_api_key_path, 'r') as file:
        ollama_api_key = file.read().strip()
    os.environ["OLLAMA_API_KEY"] = ollama_api_key
else:
    st.error(f"Ollama API key not found at {ollama_api_key_path}")
    st.stop()
prompt = (
    "You are an assistant designed to provide concise, relevant responses based on cognitive load analysis. "
        "Avoid unnecessary details or elaborations, focusing only on essential information to minimize cognitive load for the user."
)

# Initialize the LLM model
try:
    llm = OllamaLLM(model="phi",
                    temperature=0.5,
                    max_tokens=50)
except Exception as e:
    st.error(f"Error initializing LLM model: {e}")
    st.stop()

# Define the file path for emotional state data
DATA_STORAGE_DIR = os.path.dirname(__file__)
dashboard_results_path = os.path.join(DATA_STORAGE_DIR, "dashboard_results.txt")


def get_emotional_state_and_load():  # THIS IS IMPORTED FUNCTION FROM DASHBOARD
    """Reads emotional state and cognitive load from the file_save_gsr created by the dashboard."""
    try:
        with open(dashboard_results_path, "r") as file_save_gsr:
            emotional_state1 = "Unknown"
            cognitive_load1 = 0.0
            for line in file_save_gsr:
                if "Emotional State:" in line:
                    emotional_state1 = line.split(":")[1].strip()
                elif "Percentage of time under high cognitive load:" in line:
                    cognitive_load1 = float(line.split(":")[1].strip().replace("%", ""))
            return emotional_state1, cognitive_load1
    except FileNotFoundError:
        return "Unknown", 0.0  # Default if the file_save_gsr doesn't exist or hasn't been created yet


# Get the current emotional state and cognitive load
emotional_state, cognitive_load = get_emotional_state_and_load()


def run_chatbot():
    st.title("Avicenna")

    llm1 = OllamaLLM(model="phi",
                     temperature=0.1,
                     max_tokens=50)
    # Display previous results from the dashboard
    if os.path.exists(dashboard_results_path):
        with open(dashboard_results_path, "r") as f:
            dashboard_results = f.read()
    else:
        dashboard_results = "No emotional state data available from the dashboard."

    st.write("Analysis Result from Dashboard:")
    st.write(dashboard_results)

    # User input and responseChatBot generation
    user_emotion_gsr = st.text_input("How are you feeling?")
    if st.button("Get Guidance"):
        # Craft the prompt1 based on emotional state and cognitive load
        if cognitive_load > 50:  # High cognitive load threshold
            prompt1 = f"{dashboard_results} User feeling: {user_emotion_gsr}. Current emotional state: {emotional_state}." \
                     f"You are experiencing high cognitive load. Suggest ways to manage stress and improve focus."
        elif cognitive_load > 20:  # Moderate cognitive load threshold
            prompt1 = f"{dashboard_results} User feeling: {user_emotion_gsr}. Current emotional state: {emotional_state}." \
                     f"You are experiencing moderate cognitive load. Suggest strategies to maintain productivity."
        else:  # Low cognitive load
            prompt1 = f"{dashboard_results} User feeling: {user_emotion_gsr}. Current emotional state: {emotional_state}."\
                     f"Your cognitive load is low. Suggest ways to stay engaged and productive."

        try:
            responseChatBot = llm1.invoke(prompt1)
            st.write("Guidance:")
            st.write(responseChatBot)
        except Exception as ee:
            st.error(f"Error generating guidance: {ee}")


# Initialize Chroma DB with embeddings
try:
    embeddings = OllamaEmbeddings(model="phi")
    db = Chroma(persist_directory='chroma_db', embedding_function=embeddings)
except Exception as e:
    st.error(f"Error initializing Chroma DB: {e}")
    st.stop()

# Chatbot interaction with emotional state integration
st.title("Avicenna - Emotional Support Chatbot")
st.write(
    "Reflect on how you're feeling. Are you experiencing sadness, anxiety, or difficulty concentrating? Type your "
    "feelings below:")

# User input for current emotion
user_emotion = st.text_input("What are you feeling?")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, low_memory=False)  # Adjust dtype as necessary
        chunk_size = 100
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            for _, row in chunk.iterrows():
                db.add_document(doc=row.to_dict())
    except Exception as e:
        st.error(f"Error reading uploaded CSV file: {e}")
        st.stop()
else:
    st.warning("Please upload a CSV file.")

# Generate response if input is provided
if st.button("Get Guidance"):
    # Craft the prompt based on emotional state and cognitive load
    if cognitive_load > 50:  # High cognitive load threshold
        prompt2 = f"{prompt}The user expresses: {user_emotion}. Current emotional state: {emotional_state}. " \
                  f"You are experiencing high cognitive load. Suggest ways to manage stress and improve focus."
    elif cognitive_load > 20:  # Moderate cognitive load threshold
        prompt2 = f"The user expresses: {user_emotion}. Current emotional state: {emotional_state}. " \
                  f"You are experiencing moderate cognitive load. Suggest strategies to maintain productivity."
    else:  # Low cognitive load
        prompt2 = f"The user expresses: {user_emotion}. Current emotional state: {emotional_state}. " \
                  f"Your cognitive load is low. Suggest ways to stay engaged and productive."

    try:
        response = llm.invoke(prompt)
        st.write("Guidance:")
        st.write(response)
    except Exception as e:
        st.error(f"Error generating guidance: {e}")
else:
    st.warning("Please enter your feelings to receive guidance.")
