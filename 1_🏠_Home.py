# 1_🏠_Home.py
import streamlit as st
import pandas as pd
import joblib

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# --- LOAD THE SAVED MODEL ---
try:
    model = joblib.load('rf_model.pkl') # We will rename the model file later
except FileNotFoundError:
    st.error("Model file not found. Please train the models first.")
    st.stop()

# --- APP TITLE AND DESCRIPTION ---
st.title('🏡 Advanced House Price Predictor')
st.markdown("This app uses a Random Forest model to predict house prices. Use the sidebar to input features.")

# --- SIDEBAR FOR USER INPUT ---
st.sidebar.header('Input House Features')
df_sample = pd.read_csv('USA_Housing.csv').head(1) # Load a sample for min/max values

# Function to create sliders
def user_input_features():
    avg_area_income = st.sidebar.slider('Average Area Income ($)', 50000, 100000, 75000)
    avg_area_house_age = st.sidebar.slider('Average House Age (years)', 1, 10, 5)
    avg_area_num_rooms = st.sidebar.slider('Average Number of Rooms', 2, 10, 6)
    area_population = st.sidebar.slider('Area Population', 10000, 70000, 35000)
    
    data = {'Avg. Area Income': avg_area_income,
            'Avg. Area House Age': avg_area_house_age,
            'Avg. Area Number of Rooms': avg_area_num_rooms,
            'Avg. Area Number of Bedrooms': 4, # Using a default value
            'Area Population': area_population}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- DISPLAY PREDICTION ---
st.header("Prediction")
prediction = model.predict(input_df)

st.metric(label="Predicted House Price", value=f"${prediction[0]:,.2f}")

st.markdown("---")
st.write("The input features you selected are displayed below:")
st.dataframe(input_df, use_container_width=True)