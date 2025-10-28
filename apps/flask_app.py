from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)
MODEL_DIR = "../model/model.pkl"

if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError("Model not found. Please run train.py first")

model = joblib.load(MODEL_DIR)

@app.route("/")
def home():
    return {"message": "Customer Churn Prediction API"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    print(data)
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0, 1]
    return jsonify({"prediction": int(pred), "probability": float(proba)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

