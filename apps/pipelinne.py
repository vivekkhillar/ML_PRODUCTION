import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

def build_pipeline(numerical_features, categorical_features):
    """
    Build a consistent ML pipeline with preprocessing and model training.
    Locks feature order and metadata for reliable loading and prediction.
    """

    # Numeric preprocessing: handle missing values + scale
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical preprocessing: impute missing + one-hot encode
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # Final pipeline with classifier
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Attach feature metadata for future validation
    pipeline.feature_metadata = {
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "all_features": numerical_features + categorical_features
    }

    return pipeline
