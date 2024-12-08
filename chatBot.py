import os
import streamlit as st
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM

# Explicitly point to the .env file
dotenv_path = "venv\\api_keys.env"
load_dotenv(dotenv_path=dotenv_path)

# Configuration
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")  # Retrieve LANGSMITH_API_KEY from .env
OLLAMA_API_KEY_PATH = os.getenv("OLLAMA_API_KEY_PATH")  # Retrieve OLLAMA_API_KEY_PATH from .env

# Check if the API keys are set correctly
if LANGSMITH_API_KEY is None:
    st.error("LANGSMITH_API_KEY is not set. Please provide your API key.")
    st.stop()

if OLLAMA_API_KEY_PATH is None:
    st.error("OLLAMA_API_KEY_PATH is not set. Please provide the path to your Ollama API key.")
    st.stop()

# Set API keys in environment variables
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY

# Check and read Ollama API key if file exists
if os.path.exists(OLLAMA_API_KEY_PATH):
    with open(OLLAMA_API_KEY_PATH, 'r') as file:
        os.environ["OLLAMA_API_KEY"] = file.read().strip()
else:
    st.error(f"Ollama API key file not found at {OLLAMA_API_KEY_PATH}")
    st.stop()

# Model Configuration
MODEL_COGNITIVE_LOAD = "mistral"
MODEL_USER_SPECIFIC = "llama3.2"

# File paths for results
DATA_STORAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_RESULTS_PATH = r"C:\Users\dtfygu876\prompt_codes\csvChunking\Chatbot_for_Biosensor\emotional_state_logs"
EMOTIONAL_STATE_FILE = os.path.join(DASHBOARD_RESULTS_PATH, "emotional_state.txt")

ECG_RESULTS_FILE = os.path.join(DATA_STORAGE_DIR,
                                r"C:\Users\dtfygu876\prompt_codes\csvChunking\Chatbot_for_Biosensor\uploaded_data_ecg\ecgAnalysisSteps.txt")
EMG_RESULTS_FILE = os.path.join(DATA_STORAGE_DIR, "emgAnalysisSteps.txt")

# Initialize models once
try:
    llm_cognitive_load = OllamaLLM(model=MODEL_COGNITIVE_LOAD, temperature=0.5, max_tokens=50
                                   )
    llm_user_specific = OllamaLLM(model=MODEL_USER_SPECIFIC, temperature=0.5, max_tokens=50
                                  )
except Exception as e:
    st.error(f"Error initializing the models: {e}")
    st.stop()


@st.cache_data
def get_emotional_state():
    """Reads the emotional state from the dashboard results and saves it to a text file."""
    try:
        if not os.path.isdir(DASHBOARD_RESULTS_PATH):
            os.makedirs(DASHBOARD_RESULTS_PATH, exist_ok=True)

        emotional_state = "Unknown"  # Default value
        if os.path.exists(EMOTIONAL_STATE_FILE):
            with open(EMOTIONAL_STATE_FILE, "r") as file:
                lines = file.readlines()
                emotional_state = lines[-1].strip() if lines else "No data available."

        # Write or update emotional state
        with open(EMOTIONAL_STATE_FILE, "w") as output_file:
            output_file.write(f"Emotional State: {emotional_state}")

        return emotional_state
    except Exception as e:
        st.error(f"Error processing emotional state: {e}")
        return "Error reading emotional state."


@st.cache_data
def get_stress_level_from_ecg():
    """Reads the entire last line and returns the stress level in the correct format."""
    try:
        if not os.path.exists(ECG_RESULTS_FILE):
            return "ECG analysis results file not found."

        with open(ECG_RESULTS_FILE, "r") as file:
            # Read all lines and get the last one
            lines = file.readlines()
            if lines:
                return lines[-1].strip()  # Return the last line
            else:
                return "ECG results file is empty."
    except Exception as e:
        return f"An error occurred while reading the ECG results: {e}"


@st.cache_data
def get_muscle_activation_from_emg():
    """Reads muscle activation level and explanation from the EMG analysis results file."""
    try:
        if not os.path.exists(EMG_RESULTS_FILE):
            return "EMG analysis results file not found.", ""

        high_activation_percentage = "N/A"
        explanation = ""

        with open(EMG_RESULTS_FILE, "r") as file:
            for line in file:
                if "Percentage of time under high muscle activation:" in line:
                    high_activation_percentage = line.split(":")[1].strip()
                elif "Explanation of Muscle Activation Levels" in line:
                    explanation = line.strip() + "\n" + "\n".join(file.readlines())

        return high_activation_percentage, explanation
    except Exception as e:
        return f"An error occurred while reading the EMG results: {e}", ""


def create_cognitive_load_prompt(emotional_state, stress_level, high_activation_percentage, emg_explanation):
    """Generates a prompt based on cognitive load, emotional state, stress level, and EMG activation."""
    return (
        f"Analyze the following inputs and provide professional advice: \n"
        f"1. Emotional state: '{emotional_state}'\n"
        f"2. Stress level: {stress_level}%\n"
        f"3. Muscle activation intensity: {high_activation_percentage}%\n"
        f"4. EMG explanation: {emg_explanation}\n\n"
        f"Generate cognitive load percentage and generate response. send to user as:Estimated cognitive load: cognitive_load%.\n"
        f"Provide step-by-step reasoning for this estimate and suggest actions to improve the student's well-being."
    )


def create_user_specific_prompt(user_input, emotional_state, stress_level):
    """Generates a user-specific response prompt."""
    return (
        f"The user has provided the following input: '{user_input}'.\n"
        f"Task:\n"
        f"1. If the query is about emotional state, use {get_emotional_state()} to analyze and provide a detailed explanation.\n"
        f"2. If the query relates to stress level, analyze {get_stress_level_from_ecg()} and offer insights.\n\n"
        f"Examples:\n"
        f"- User: 'I feel anxious and cannot concentrate.'\n"
        f"  Response: 'Your emotional state indicates anxiety. Here are some techniques to reduce stress and improve focus...'\n"
        f"- User: 'What is my current stress level?'\n"
        f"  Response: 'Your stress level based on ECG data is moderate. This may affect your cognitive load. Here’s what you can do to manage it...'\n\n"
        f"Based on these examples, generate an appropriate response to assist the user effectively."
    )


def run_chatbot():
    st.subheader("Chat with Avicenna")

    # Cached dashboard results
    emotional_state = get_emotional_state()

    # Get ECG stress level
    stress_level = get_stress_level_from_ecg()

    # Get EMG muscle activation level
    high_activation_percentage, emg_explanation = get_muscle_activation_from_emg()

    # Display results in expanders
    with st.expander("Emotional State Results"):
        st.write(f"Emotional State: {emotional_state}")

    with st.expander("ECG Stress Level"):
        st.write(stress_level)

    with st.expander("EMG Muscle Activation"):
        st.write(f"Percentage of time under high muscle activation: {high_activation_percentage}")
        st.write(emg_explanation)

    # Cognitive load response generation
    cognitive_load_prompt = create_cognitive_load_prompt(emotional_state, stress_level, high_activation_percentage,
                                                         emg_explanation)
    try:
        cognitive_response = llm_cognitive_load.invoke(cognitive_load_prompt)
        with st.container():
            st.subheader("Analysis Suggestion")
            st.write(cognitive_response)
    except Exception as e:
        st.error(f"Error generating response: {e}")

    # Chat interaction
    with st.container():
        st.subheader("Chat History")
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        for entry in st.session_state["chat_history"]:
            st.write(f"**User:** {entry['user']}")
            st.write(f"**Bot:** {entry['bot']}")

    # User input
    user_input = st.text_input("How are you feeling or any specific issue you'd like to discuss?", key="user_input")

    if st.button("Send"):
        if user_input:
            user_prompt = create_user_specific_prompt(user_input, emotional_state, stress_level)
            try:
                user_response = llm_user_specific.invoke(user_prompt)
                st.session_state["chat_history"].append({"user": user_input, "bot": user_response})
                st.write("Avicenna's Response to Your Message:")
                st.write(user_response)
            except Exception as e:
                st.error(f"Error generating response for user input: {e}")
        else:
            st.warning("Please enter your feelings or a specific question.")


if __name__ == "__main__":
    run_chatbot()