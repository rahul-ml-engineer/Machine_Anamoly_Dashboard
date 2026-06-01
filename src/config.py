from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "Data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "Data" / "processed"
STREAMING_DATA_DIR = BASE_DIR / "Data" / "streaming"

TRAIN_DATA_PATH = list(RAW_DATA_DIR.rglob("*.csv"))
PROCESSED_TRAIN_PATH = PROCESSED_DATA_DIR / "train_processed.csv"
FEATURES_TRAIN_PATH = PROCESSED_DATA_DIR / "train_features.csv"

Anamoly_RESULTS_PATH = PROCESSED_DATA_DIR / "anomaly_results.csv"
STREAMING_RESULTS_PATH = STREAMING_DATA_DIR / "streaming_results.csv"
FINAL_STREAMING_RESULTS_PATH = STREAMING_DATA_DIR / "streaming_results_explained.csv"
MODEL_DIR = BASE_DIR / "models"
ISOLATION_FOREST_MODEL_PATH = MODEL_DIR / "isolation_forest.pkl"
SCALER_MODEL_PATH = MODEL_DIR / "scaler.pkl"