# pages/3_🤖_Model_Comparison.py
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Model Comparison", page_icon="🤖", layout="centered")

st.title('🤖 Model Comparison')
st.markdown("Compare predictions from different machine learning models.")

# --- LOAD MODELS ---
try:
    lr_model = joblib.load('linear_model.pkl')
    rf_model = joblib.load('rf_model.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please train the models first.")
    st.stop()

# --- USER INPUT ---
st.sidebar.header('Input House Features')
# Function to create sliders
def user_input_features():
    avg_area_income = st.sidebar.slider('Average Area Income ($)', 50000, 100000, 75000)
    avg_area_house_age = st.sidebar.slider('Average House Age (years)', 1, 10, 5)
    avg_area_num_rooms = st.sidebar.slider('Average Number of Rooms', 2, 10, 6)
    area_population = st.sidebar.slider('Area Population', 10000, 70000, 35000)
    
    data = {'Avg. Area Income': avg_area_income,
            'Avg. Area House Age': avg_area_house_age,
            'Avg. Area Number of Rooms': avg_area_num_rooms,
            'Avg. Area Number of Bedrooms': 4,
            'Area Population': area_population}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- DISPLAY COMPARISON ---
st.header("Prediction Comparison")

# Make predictions
lr_prediction = lr_model.predict(input_df)
rf_prediction = rf_model.predict(input_df)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Linear Regression")
    st.metric(label="Predicted Price", value=f"${lr_prediction[0]:,.2f}")

with col2:
    st.subheader("Random Forest")
    st.metric(label="Predicted Price", value=f"${rf_prediction[0]:,.2f}")

st.markdown("---")
st.info("**Note:** Random Forest models can capture more complex, non-linear patterns in the data, which often leads to more accurate (and different) predictions compared to Linear Regression.")