import pandas as pd
import numpy as np
import os
import pickle
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute  import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,accuracy_score
from data_loader import data_load
from preprocess import preprocess_data
from pipelinne import build_pipeline

DATA_PATH = '../data/customer_churn.csv'
MODEL_DIR = f"{os.path.dirname(__file__)}/../model"

def main():

    df = data_load()

    df = preprocess_data(df)

    # features and target variable

    x = df.drop(columns=['customerID','Churn'])
    y = df['Churn']

    numerical_features = x.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_features = x.select_dtypes(include=['object','category']).columns.tolist()

    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y) 

    pipeline = build_pipeline(numerical_features,categorical_features) 

    pipeline.fit(X_train,y_train)
    preds  = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:,1]

    report = classification_report(y_test,preds)
    print(report)
    cm = confusion_matrix(y_test,preds)
    auc = roc_auc_score(y_test,proba)

    print("Accuracy:", accuracy_score(y_test, preds)*100)
    print("AUC:", auc)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "model.pkl"))
    print("Model saved to", os.path.join(MODEL_DIR, "model.pkl"))

if __name__ == "__main__":
    main()