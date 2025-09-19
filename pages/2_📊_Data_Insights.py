# pages/2_📊_Data_Insights.py
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Data Insights", page_icon="📊", layout="wide")

st.title('📊 Data Insights and Visualizations')

# --- LOAD DATA AND MODEL ---
try:
    df = pd.read_csv('USA_Housing.csv')
    model = joblib.load('rf_model.pkl')
except FileNotFoundError:
    st.error("Required files not found. Please run the training script and ensure data is present.")
    st.stop()

# --- FEATURE IMPORTANCE PLOT ---
st.header('Feature Importance')
st.markdown("This chart shows which features are most influential in the Random Forest model's predictions.")

# Extract feature importances
importances = model.feature_importances_
feature_names = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis', ax=ax)
ax.set_title('Feature Importance for House Price Prediction')
st.pyplot(fig)


st.markdown("---")

# --- INTERACTIVE MAP ---
st.header('Interactive Map of Housing Data')
st.markdown("Since the dataset doesn't include coordinates, we've generated random latitude and longitude values for demonstration purposes.")

# Generate random coordinates for visualization
np.random.seed(42)
df['latitude'] = np.random.uniform(32, 40, size=len(df))
df['longitude'] = np.random.uniform(-120, -80, size=len(df))

st.map(df[['latitude', 'longitude']])