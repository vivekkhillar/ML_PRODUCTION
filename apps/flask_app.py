from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)
MODEL_PATH = f"{os.path.dirname(__file__)}/../model/model.pkl"

# --- Load Model ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(" Model not found. Please run train.py first.")

model = joblib.load(MODEL_PATH)
print(model)
print(" Model loaded successfully!")

# Check feature metadata
if hasattr(model, "feature_metadata"):
    feature_metadata = model.feature_metadata
    expected_features = feature_metadata["all_features"]
    print(" Expected features:", expected_features)
else:
    raise AttributeError("Model missing feature metadata. Please retrain using updated train.py.")


@app.route("/")
def home():
    return {"message": "Customer Churn Prediction API is running 🚀"}


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse incoming JSON
        data = request.get_json(force=True)
        df = pd.DataFrame([data])

        #--- Validation and Alignment ---
        for col in expected_features:
            if col not in df.columns:
                df[col] = None  # Add missing columns with None

        df = df[expected_features]
    
        # Debug view
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("\nIncoming request dataframe:\n", df)

        # --- Predict ---
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0, 1]

        return jsonify({
            "prediction": pred,
            "probability": proba
        })

    except Exception as e:
        print(" Prediction error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
