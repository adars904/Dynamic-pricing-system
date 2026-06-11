from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import numpy as np
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load all models at startup
with open("models/dt_model.pkl", "rb") as f:
    dt_model = pickle.load(f)

with open("models/lr_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

tf_model = tf.keras.models.load_model("models/tf_model.h5", compile=False)


class PriceInput(BaseModel):
    rating: float
    review_count: int
    base_price: float
    discount_percentage: float
    hour: int
    day_of_week: int


@app.post("/predict")
def predict(data: PriceInput):
    demand_level = 2 if data.rating >= 4.5 else (1 if data.rating >= 3.5 else 0)
    is_peak_hour = 1 if 17 <= data.hour <= 21 else 0
    is_weekend   = 1 if data.day_of_week >= 5 else 0

    features = np.array([[
        demand_level,
        data.review_count,
        data.hour,
        data.day_of_week,
        data.base_price,
        is_peak_hour,
        is_weekend,
        data.discount_percentage
    ]])

    scaled = scaler.transform(features)

    dt_price = float(dt_model.predict(scaled)[0])
    lr_price = float(lr_model.predict(scaled)[0])
    nn_price = float(tf_model.predict(scaled, verbose=0).flatten()[0])
    ensemble = round((dt_price + lr_price + nn_price) / 3, 2)

    return {
        "decision_tree":     round(dt_price, 2),
        "linear_regression": round(lr_price, 2),
        "neural_network":    round(nn_price, 2),
        "ensemble":          ensemble,
        "demand_level":      demand_level,
        "is_peak_hour":      bool(is_peak_hour),
        "is_weekend":        bool(is_weekend),
    }


@app.get("/health")
def health():
    return {"status": "ok"}

# Serve the frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")