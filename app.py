import streamlit as st
import pandas as pd
import joblib

# Set the page configuration
st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="centered")

# Load your trained model
model = joblib.load('house_price_model.pkl')

# Create the title and description
st.title("🏡 House Price Predictor")
st.write("Enter the details of a house to predict its price.")

# Create input fields in the sidebar
st.sidebar.header("Input Features")
avg_area_income = st.sidebar.slider("Average Area Income", 50000, 100000, 75000)
avg_area_house_age = st.sidebar.slider("Average House Age", 1, 10, 5)
avg_area_num_rooms = st.sidebar.slider("Average Number of Rooms", 2, 10, 6)
area_population = st.sidebar.slider("Area Population", 10000, 70000, 35000)

# Create a button to make predictions
if st.sidebar.button("Predict Price"):
    # Prepare the input data for the model
    input_data = pd.DataFrame({
        'Avg. Area Income': [avg_area_income],
        'Avg. Area House Age': [avg_area_house_age],
        'Avg. Area Number of Rooms': [avg_area_num_rooms],
        # Note: We add a default for bedrooms as it's in the model but not a slider
        'Avg. Area Number of Bedrooms': [4],
        'Area Population': [area_population]
    })

    # Make the prediction
    prediction = model.predict(input_data)

    # Display the result
    st.success(f"The predicted house price is ${prediction[0]:,.2f}")