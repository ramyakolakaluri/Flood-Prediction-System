"""
Rising Waters — Flask web application.

Loads the trained model + scaler (produced by train_model.py) and serves
a simple web UI where a user can enter current rainfall and weather
readings to get a real-time flood risk prediction.
"""

import os

import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = "models/flood_model.joblib"
SCALER_PATH = "models/scaler.joblib"
FEATURES_PATH = "models/feature_order.joblib"

model = None
scaler = None
feature_order = None


def load_artifacts():
    global model, scaler, feature_order
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_order = joblib.load(FEATURES_PATH)


load_artifacts()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template(
            "index.html",
            prediction=None,
            error="Model not found. Run `python train_model.py` first.",
        )

    try:
        values = [float(request.form[feat]) for feat in feature_order]
    except (KeyError, ValueError):
        return render_template(
            "index.html", prediction=None, error="Please fill in all fields with valid numbers."
        )

    X = np.array(values).reshape(1, -1)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1] if hasattr(model, "predict_proba") else None

    result = {
        "risk": "Flood Risk" if pred == 1 else "Low Risk",
        "is_risk": bool(pred),
        "confidence": round(float(proba) * 100, 1) if proba is not None else None,
    }

    return render_template("index.html", prediction=result, form_values=request.form)


if __name__ == "__main__":
    app.run(debug=True)
