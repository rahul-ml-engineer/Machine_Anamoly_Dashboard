from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "Data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "Data" / "processed"

TRAIN_DATA_PATH = list(RAW_DATA_DIR.rglob("*.csv"))
PROCESSED_TRAIN_PATH = PROCESSED_DATA_DIR / "train_processed.csv"
FEATURES_TRAIN_PATH = PROCESSED_DATA_DIR / "train_features.csv"
