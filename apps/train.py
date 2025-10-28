import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from data_loader import data_load
from preprocess import preprocess_data
from pipelinne import build_pipeline  #  note spelling fix

DATA_PATH = '../data/customer_churn.csv'
MODEL_DIR = f"{os.path.dirname(__file__)}/../model"

def main():
    # Load and preprocess data
    df = data_load()
    df = preprocess_data(df)

    # Define features and target
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']

    numerical_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [
        'gender', 'Partner', 'Dependents', 'PhoneService',
        'InternetService', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'tenure_group'
    ]

    print("\nNUMERICAL FEATURES:", numerical_features)
    print("CATEGORICAL FEATURES:", categorical_features)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build pipeline
    pipeline = build_pipeline(numerical_features, categorical_features)

    # Fit model
    pipeline.fit(X_train, y_train)
    print("\n Model training complete!")

    # Evaluate
    preds = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    print("\nClassification Report:\n", report)
    print("Accuracy:", round(accuracy_score(y_test, preds)*100, 2))
    print("AUC:", round(auc, 4))

    # --- Save Model with Feature Metadata ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "model.pkl")

    # Save pipeline
    joblib.dump(pipeline, model_path)

    # Load it again and attach metadata
    model = joblib.load(model_path)
    model.feature_metadata = {
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "all_features": numerical_features + categorical_features
    }

    joblib.dump(model, model_path)
    print(f"\n Model saved successfully with feature metadata at: {model_path}")

if __name__ == "__main__":
    main()
