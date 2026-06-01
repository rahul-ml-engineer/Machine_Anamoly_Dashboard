import pandas as pd
import numpy as np
import time
import joblib

from config import ISOLATION_FOREST_MODEL_PATH, SCALER_MODEL_PATH, Anamoly_RESULTS_PATH, STREAMING_RESULTS_PATH

def load_models():
    model = joblib.load(ISOLATION_FOREST_MODEL_PATH)
    scaler = joblib.load(SCALER_MODEL_PATH)
    return model, scaler

def load_results():
    stream_df = pd.read_csv(Anamoly_RESULTS_PATH)
    return stream_df

def main():
    model, scaler = load_models()
    stream_df = load_results()
    X = stream_df.drop(["datetime", "fault_type", "source_file", "anomaly", "anomaly_score"], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.bfill()

    stream_results = []

    for i in range(len(X)):
        row = X.iloc[[i]]
        row_scaled = scaler.transform(row)
        prediction = model.predict(row_scaled)[0]
        score = model.decision_function(row_scaled)[0]
        stream_results.append({"index": i, "prediction": prediction, "anomaly_score": score})

        if prediction == -1:
            print(f"⚠ ALERT at row {i} | "f"Score: {score:.4f}")

        time.sleep(0.01)

    results_df = pd.DataFrame(stream_results)
    results_df["status"] = results_df["prediction"].map({1: "Normal", -1: "Anomaly"})
    results_df.to_csv(STREAMING_RESULTS_PATH, index=False)

    print("\nStreaming completed!")
    print(results_df.head())

    anomalies = results_df[results_df["prediction"] == -1]
    print("\nDetected Anomalies:")
    print(anomalies.head())


if __name__ == "__main__":
    main()