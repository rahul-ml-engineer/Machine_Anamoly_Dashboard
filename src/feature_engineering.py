import pandas as pd
from config import PROCESSED_TRAIN_PATH, FEATURES_TRAIN_PATH

def load_data():
    df = pd.read_csv(PROCESSED_TRAIN_PATH)
    return df

def feature_engineering(df):
    sensor_cols = [col for col in df.columns if col not in ["datetime", "fault_type", "source_file"]]
    feature_df = df.copy()
    
    window_size = 50

    for col in sensor_cols:
        feature_df[f"{col}_rolling_mean"] = (feature_df[col].rolling(window=window_size).mean())
        feature_df[f"{col}_rolling_std"] = (feature_df[col].rolling(window=window_size).std())
        feature_df[f"{col}_lag1"] = (feature_df[col].shift(1))
        feature_df[f"{col}_diff"] = (feature_df[col].diff())
        feature_df[f"{col}_roc"] = (feature_df[col].pct_change())
        mean = feature_df[col].mean()
        std = feature_df[col].std()
        feature_df[f"{col}_zscore"] = ((feature_df[col] - mean) / std)
    
    return feature_df

def data_cleaning(feature_df):
    feature_df = feature_df.copy()
    feature_df = feature_df.bfill()
    return feature_df

def save_features(feature_df):
    FEATURES_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(FEATURES_TRAIN_PATH, index=False)

def main():
    df = load_data()
    feature_df = feature_engineering(df)
    feature_df = data_cleaning(feature_df)
    save_features(feature_df)
    print("Original Shape:", df.shape)
    print("Feature Engineered Shape:", feature_df.shape)
    print("Feature engineering completed successfully")

if __name__ == "__main__":
    main()