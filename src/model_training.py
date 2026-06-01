from xml.parsers.expat import model

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from config import FEATURES_TRAIN_PATH, Anamoly_RESULTS_PATH, ISOLATION_FOREST_MODEL_PATH, SCALER_MODEL_PATH


def load_data():
    feature_df = pd.read_csv(FEATURES_TRAIN_PATH)
    return feature_df


def preprocess_data(feature_df):
    X = feature_df.drop(["datetime", "fault_type", "source_file"], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.bfill()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def train_model(X_scaled):
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(X_scaled)
    return model

def detect_anomalies(feature_df, model, X_scaled):
    predictions = model.predict(X_scaled)
    feature_df["anomaly"] = predictions
    scores = model.decision_function(X_scaled)
    feature_df["anomaly_score"] = scores
    return feature_df


def save_results(feature_df, model, scaler):
    Anamoly_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ISOLATION_FOREST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCALER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(Anamoly_RESULTS_PATH, index=False)
    joblib.dump(model, ISOLATION_FOREST_MODEL_PATH)
    joblib.dump(scaler, SCALER_MODEL_PATH)

def main():
    feature_df = load_data()
    X_scaled, scaler = preprocess_data(feature_df)
    model = train_model(X_scaled)
    feature_df = detect_anomalies(feature_df, model, X_scaled)
    save_results(feature_df, model, scaler)

    print("Anomaly detection completed successfully!")
    print(feature_df[["anomaly", "anomaly_score"]].head())


if __name__ == "__main__":
    main()