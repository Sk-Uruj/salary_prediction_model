
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- Custom CSS for Glassmorphism Theme ---
st.markdown("""
<style>
/* Basic background for glass effect */
.stApp {
    background-image: linear-gradient(to right top, #d16ba5, #c777b9, #ba83ca, #aa8fd8, #9a9ae1, #8aa7ec, #79b3ed, #69bff8, #52cffe, #41dfff, #46eefa, #5ffbf1);
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* Apply glassmorphism to Streamlit's main content block */
div[data-testid="stVerticalBlock"] > div:first-child {
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    padding: 20px;
    margin-bottom: 20px;
}

/* Styling for sidebar if you use one */
.st-emotion-cache-1ldf15n {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
}

/* Further refine elements for glass effect */
.stButton>button, .stSelectbox>div>div, .stSlider>div>div {
    background: rgba(255, 255, 255, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(5px);
    -webkit-backdrop-filter: blur(5px);
    border-radius: 8px;
    color: #333;
    font-weight: bold;
}

.stButton>button:hover {
    background: rgba(255, 255, 255, 0.4);
    border-color: rgba(255, 255, 255, 0.4);
}

h1, h2, h3, h4, h5, h6, .stMarkdown, .stText, .stAlert {
    color: #333;
    text-shadow: 1px 1px 2px rgba(255,255,255,0.7);
}

</style>
""", unsafe_allow_html=True)

# --- Load the trained model ---
# Ensure 'best_model.pkl' is in the same directory as this script
try:
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: 'best_model.pkl' not found. Please ensure the model file is in the correct directory.")
    st.stop()

# Define the mapping for ExperienceLevel (should be the same as used during training)
# Based on the previous notebook execution: ['Junior', 'Mid', 'Senior'] -> [0, 1, 2]
experience_level_mapping = {'Junior': 0, 'Mid': 1, 'Senior': 2}

# --- Streamlit App Layout ---
st.title('Salary Prediction App 💰')
st.write('Welcome to the Glassmorphism Salary Predictor!')
st.markdown('Adjust the parameters below to get an estimated salary based on our trained machine learning model.')

# Input Widgets
st.header('Input Parameters')

# User input for YearsExperience
years_experience = st.slider(
    'Years of Experience',
    min_value=0.0, 
    max_value=20.0, 
    value=5.0, 
    step=0.1,
    help='Drag to select the number of years of professional experience.'
)

# User input for ExperienceLevel
experience_level = st.selectbox(
    'Experience Level',
    options=['Junior', 'Mid', 'Senior'],
    index=1, # Default to 'Mid'
    help='Select your general experience level.'
)

# Map the selected experience level to its encoded numerical value
experience_level_encoded = experience_level_mapping[experience_level]

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    'YearsExperience': [years_experience],
    'ExperienceLevel_Encoded': [experience_level_encoded]
})

# Prediction Button
if st.button('Predict Salary 🚀'):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f'### Predicted Salary: ${prediction:,.2f}')
        st.balloons()
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("""
---
<small>Developed with Streamlit and a touch of Glassmorphism magic.</small>
""")
