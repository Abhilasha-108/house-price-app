# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
# Import RandomForestRegressor instead of LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Load the dataset
print("Loading data...")
df = pd.read_csv('USA_Housing.csv')

# 2. Prepare the data
features = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']
target = 'Price'
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Choose and train the new model
# We are now using RandomForestRegressor with 100 decision trees
print("Training Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# 4. Save the trained model (this will overwrite the old one)
joblib.dump(model, 'house_price_model.pkl')
print("New Random Forest model saved to house_price_model.pkl")