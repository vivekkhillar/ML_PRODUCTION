import numpy as np
import pandas as pd
import os
from sklearn import pipeline
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    # Convert 'TotalCharges' to numeric, coerce errors to NaN, then fill NaN with 0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # convert the yes or no values to 0 and 1

    for col in ['Partner','Dependents','PhoneService','PaperlessBilling','Churn']:
        df[col] = df[col].map({'yes': 1, 'no' : 0, 'Yes': 1, 'No' : 0})

    # encode the Gender column to 0 and 1 by map

    df['gender'] = df['gender'].map({'Female':0,'Male': 1})

    # convert tenure to the labels

    df['tenure_group'] = pd.cut(df['tenure'], bins=[-1,12,24,48,72], labels=['0-12','13-24','25-48','49+'])

    return df