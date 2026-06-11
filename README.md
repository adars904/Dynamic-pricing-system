# Dynamic Pricing Engine — ML & Deep Learning

> Real-time product price prediction using demand signals, behavioral features, and a benchmarked Neural Network — served via FastAPI with an interactive dashboard.

---

## Overview

Static pricing ignores demand. This project builds a full-stack, data-driven pricing engine that predicts optimal product prices in real-time based on user behavior, temporal patterns, and product-level signals — trained on a real-world Amazon dataset.

The system benchmarks three models, identifies the best performer, and serves live predictions through a REST API with an interactive frontend.

---

## Features

- **Demand-aware feature engineering** from product ratings and behavioral signals
- **Temporal features** — peak hour flags, weekend indicators
- **Model benchmarking** — Linear Regression, Decision Tree, Neural Network
- **Evaluation** using MAE & MSE for rigorous comparison
- **FastAPI backend** for real-time inference
- **Interactive JS dashboard** for live price lookups

---

## Tech Stack

| Layer | Tools |
|---|---|
| ML / DL | TensorFlow, scikit-learn |
| Backend | FastAPI, Python |
| Frontend | HTML, CSS, JavaScript |
| Data Processing | pandas, NumPy, StandardScaler |

---

## Project Structure

```
DYNAMICPRICING/
├── models/                        # Saved trained models
├── static/
│   └── index.html                 # Frontend dashboard
├── venv/                          # Virtual environment
├── .gitignore
├── amazon.csv                     # Amazon product dataset
├── main.py                        # FastAPI app & API endpoints
└── model.py                       # ML pipeline, training & inference logic
```

---

## ML Pipeline

```
Raw Data → Feature Engineering → Normalization → Model Training → Evaluation → API Serving
```

### Feature Engineering

- **Demand level** — derived from product rating counts as a behavioral proxy for purchase velocity
- **Peak hour flag** — binary feature marking high-traffic hours
- **Weekend flag** — captures weekend demand spikes
- **StandardScaler** — applied across all features for stable gradient descent

### Model Comparison

| Model | MAE | MSE |
|---|---|---|
| Linear Regression | Higher | Higher |
| Decision Tree | Medium | Medium |
| **Neural Network** | **Lowest** | **Lowest** |

The Neural Network outperforms classical models by capturing nonlinear relationships between demand signals and price — something linear models fundamentally cannot do.

---

## Neural Network Architecture

```
Input Layer
    ↓
Dense(64, activation='relu')
    ↓
Dense(32, activation='relu')
    ↓
Dense(1, activation='linear')   ← Price output
```

- **Optimizer:** Adam
- **Loss:** Mean Squared Error (MSE)
- **Training:** Backpropagation on normalized feature vectors
- Two hidden layers — additional depth showed no meaningful error reduction on this dataset

---

## API

### Run the server

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

### Predict endpoint

```
POST /predict
```

**Request body:**
```json
{
  "rating_count": 1500,
  "is_peak_hour": 1,
  "is_weekend": 0
}
```

**Response:**
```json
{
  "predicted_price": 42.75
}
```

The model is loaded once at startup — not per request — keeping inference latency minimal.

---

## Frontend Dashboard

Open `static/index.html` or point it to your running FastAPI instance (`http://127.0.0.1:8000`). The dashboard lets you input demand signals and get a real-time predicted price back from the model.

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/your-username/dynamic-pricing-engine.git
cd DYNAMICPRICING

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn tensorflow scikit-learn pandas numpy

# Train the model (saves to /models)
python model.py

# Start the API
uvicorn main:app --reload
```

---

## Results

The Neural Network achieved significantly lower MAE and MSE compared to Linear Regression and Decision Tree, confirming that deep learning captures the complex, nonlinear demand-price relationships that classical models miss.

---

## Future Improvements

- Add competitor price signals as input features
- Benchmark LightGBM as a strong non-DL baseline
- Implement A/B testing for pricing strategies
- Move to online feature scaling for production use
- Containerize with Docker for deployment

---

## License

MIT License — free to use, modify, and distribute.
