import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib


def train_accident_model(data_path, model_dir="models"):
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Load the data
    df = pd.read_csv(data_path)

    # Drop rows where target is missing
    df = df.dropna(subset=["crash_type"])

    # Handle missing values
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    # Encode categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Define features and target
    X = df.drop(["crash_type", "crash_date"], axis=1, errors="ignore")
    y = df["crash_type"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Save model, feature names, and encoders
    joblib.dump(
        model,
        os.path.join(model_dir, "accident_prediction_model.pkl")
    )
    joblib.dump(
        list(X.columns),
        os.path.join(model_dir, "feature_names.pkl")
    )
    joblib.dump(
        label_encoders,
        os.path.join(model_dir, "label_encoders.pkl")
    )

    print(f"Model saved to {model_dir}/accident_prediction_model.pkl")
    print(f"Features saved to {model_dir}/feature_names.pkl")
    print(f"Label encoders saved to {model_dir}/label_encoders.pkl")
    print(f"Test accuracy: {model.score(X_test, y_test):.4f}")

    return model, label_encoders, list(X.columns)


if __name__ == "__main__":
    train_accident_model("data/traffic_accidents.csv")