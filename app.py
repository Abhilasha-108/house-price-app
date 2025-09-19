# app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Dynamic House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# --- LOAD THE SAVED MODEL AND DATA ---
model = joblib.load('house_price_model.pkl')
df = pd.read_csv('USA_Housing.csv')

# --- APP LAYOUT ---
st.title('🏡 Dynamic House Price Predictor')
st.markdown("Using a Random Forest model. The price distribution chart now updates with your prediction.")

# Create two columns
col1, col2 = st.columns([1, 2])

# --- COLUMN 1: USER INPUT ---
with col1:
    st.header("Input House Features")
    avg_area_income = st.slider('Average Area Income ($)', float(df['Avg. Area Income'].min()), float(df['Avg. Area Income'].max()), float(df['Avg. Area Income'].mean()))
    avg_area_house_age = st.slider('Average House Age (years)', float(df['Avg. Area House Age'].min()), float(df['Avg. Area House Age'].max()), float(df['Avg. Area House Age'].mean()))
    avg_area_num_rooms = st.slider('Average Number of Rooms', float(df['Avg. Area Number of Rooms'].min()), float(df['Avg. Area Number of Rooms'].max()), float(df['Avg. Area Number of Rooms'].mean()))
    area_population = st.slider('Area Population', float(df['Area Population'].min()), float(df['Area Population'].max()), float(df['Area Population'].mean()))
    
    # Store the prediction in a session state to keep the line on the chart
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None

    if st.button('Predict Price'):
        input_data = pd.DataFrame({
            'Avg. Area Income': [avg_area_income],
            'Avg. Area House Age': [avg_area_house_age],
            'Avg. Area Number of Rooms': [avg_area_num_rooms],
            'Avg. Area Number of Bedrooms': [4], # Using a default value
            'Area Population': [area_population]
        })
        
        prediction_value = model.predict(input_data)
        st.session_state.prediction = prediction_value[0]
        
        st.metric(label="Predicted House Price", value=f"${st.session_state.prediction:,.2f}")

# --- COLUMN 2: DATA VISUALIZATION ---
with col2:
    st.header("Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the histogram of all house prices
    ax.hist(df['Price'], bins=30, edgecolor='black', alpha=0.7, label='Price Distribution')
    ax.set_title("Distribution of House Prices in Dataset")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Number of Houses")
    
    # If a prediction has been made, draw a vertical line
    if st.session_state.prediction is not None:
        ax.axvline(st.session_state.prediction, color='r', linestyle='--', linewidth=2, label='Your Predicted Price')
    
    ax.legend()
    st.pyplot(fig)