import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# =========================
# 🔹 LOAD DATASET
# =========================
df = pd.read_csv("amazon.csv")

# Keep only useful columns
df = df[['product_rating', 'total_reviews', 'discounted_price', 'discount_percentage']].dropna()

# Rename for clarity
df.rename(columns={
    'product_rating': 'rating',
    'total_reviews': 'review_count',
    'discounted_price': 'base_price'
}, inplace=True)

# =========================
# 🔹 FEATURE ENGINEERING
# =========================

# Demand level from rating
df['demand_level'] = df['rating'].apply(
    lambda x: 2 if x >= 4.5 else (1 if x >= 3.5 else 0)
)

# Active users from reviews
df['active_users'] = df['review_count']

# Simulate time
df['hour'] = np.random.randint(0, 24, size=len(df))
df['day_of_week'] = np.random.randint(0, 7, size=len(df))

# Peak + weekend
df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if 17 <= x <= 21 else 0)
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# =========================
# 🔹 TARGET VARIABLE (DYNAMIC PRICE)
# =========================
def generate_price(row):
    price = row['base_price']

    # Demand effect
    if row['demand_level'] == 2:
        price *= 1.25
    elif row['demand_level'] == 0:
        price *= 0.85

    # Peak hour
    if row['is_peak_hour']:
        price *= 1.10

    # Weekend
    if row['is_weekend']:
        price *= 1.08

    # Discount effect (inverse relation)
    price *= (1 + (row['discount_percentage'] / 100) * 0.2)

    return price

df['actual_price'] = df.apply(generate_price, axis=1)

# =========================
# 🔹 FEATURES & TARGET
# =========================
X = df[['demand_level', 'active_users', 'hour', 'day_of_week',
        'base_price', 'is_peak_hour', 'is_weekend', 'discount_percentage']]

y = df['actual_price']

# =========================
# 🔹 SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 🔹 TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# 🔹 MODEL 1: Decision Tree
# =========================
dt_model = DecisionTreeRegressor(max_depth=6)
dt_model.fit(X_train, y_train)

dt_preds = dt_model.predict(X_test)

print("=== Decision Tree ===")
print("MAE:", mean_absolute_error(y_test, dt_preds))
print("MSE:", mean_squared_error(y_test, dt_preds))

# =========================
# 🔹 MODEL 2: Linear Regression
# =========================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_preds = lr_model.predict(X_test)

print("\n=== Linear Regression ===")
print("MAE:", mean_absolute_error(y_test, lr_preds))
print("MSE:", mean_squared_error(y_test, lr_preds))  
# =========================
# 🔹 MODEL 3: TensorFlow Neural Network
# =========================
tf_model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),
keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

tf_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Train
tf_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Predict
tf_preds = tf_model.predict(X_test).flatten()

print("\n=== TensorFlow Neural Network ===")
print("MAE:", mean_absolute_error(y_test, tf_preds))
print("MSE:", mean_squared_error(y_test, tf_preds))

# =========================
# 🔹 SAMPLE PREDICTION
# =========================
print("\n=== Sample Prediction ===")

sample = pd.DataFrame([[2, 150, 19, 6, 100, 1, 1, 20]],
                      columns=['demand_level', 'active_users', 'hour', 'day_of_week',
                               'base_price', 'is_peak_hour', 'is_weekend', 'discount_percentage'])

sample_scaled = scaler.transform(sample)
tf_price = tf_model.predict(sample_scaled).flatten()[0]
print("TensorFlow Price:", round(tf_price, 2))


print("Decision Tree Price:", round(dt_model.predict(sample_scaled)[0], 2))
print("Linear Regression Price:", round(lr_model.predict(sample_scaled)[0], 2))
import pickle
import os

os.makedirs("models", exist_ok=True)

# Save Decision Tree
with open("models/dt_model.pkl", "wb") as f:
    pickle.dump(dt_model, f)

# Save Linear Regression
with open("models/lr_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)

# Save Scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save TensorFlow model
tf_model.save("models/tf_model.keras")

print("✅ All models saved to /models folder")