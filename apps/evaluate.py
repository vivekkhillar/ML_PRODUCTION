import joblib
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd

def evaluate(model_path, X_test, y_test):
    model = joblib.load(model_path)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]

    report = classification_report(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    print("AUC:", auc)
    print(report)
