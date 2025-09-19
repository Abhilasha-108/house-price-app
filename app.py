# app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Advanced House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# --- LOAD THE SAVED MODEL AND DATA ---
model = joblib.load('house_price_model.pkl')
df = pd.read_csv('USA_Housing.csv')

# --- APP LAYOUT ---
st.title('🏡 Advanced House Price Predictor')
st.markdown("Using a Random Forest model for more accurate predictions.")

# Create two columns
col1, col2 = st.columns([1, 2]) 

# --- COLUMN 1: USER INPUT ---
with col1:
    st.header("Input House Features")
    avg_area_income = st.slider('Average Area Income ($)', float(df['Avg. Area Income'].min()), float(df['Avg. Area Income'].max()), float(df['Avg. Area Income'].mean()))
    avg_area_house_age = st.slider('Average House Age (years)', float(df['Avg. Area House Age'].min()), float(df['Avg. Area House Age'].max()), float(df['Avg. Area House Age'].mean()))
    avg_area_num_rooms = st.slider('Average Number of Rooms', float(df['Avg. Area Number of Rooms'].min()), float(df['Avg. Area Number of Rooms'].max()), float(df['Avg. Area Number of Rooms'].mean()))
    area_population = st.slider('Area Population', float(df['Area Population'].min()), float(df['Area Population'].max()), float(df['Area Population'].mean()))
    
    if st.button('Predict Price'):
        input_data = pd.DataFrame({
            'Avg. Area Income': [avg_area_income],
            'Avg. Area House Age': [avg_area_house_age],
            'Avg. Area Number of Rooms': [avg_area_num_rooms],
            'Avg. Area Number of Bedrooms': [4], # Using a default value
            'Area Population': [area_population]
        })
        
        prediction = model.predict(input_data)
        
        st.metric(label="Predicted House Price", value=f"${prediction[0]:,.2f}")

# --- COLUMN 2: DATA VISUALIZATION ---
with col2:
    st.header("Price Distribution")
    fig, ax = plt.subplots()
    ax.hist(df['Price'], bins=30, edgecolor='black')
    ax.set_title("Distribution of House Prices in Dataset")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Number of Houses")
    st.pyplot(fig)