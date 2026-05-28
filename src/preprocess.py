import pandas as pd
from config import TRAIN_DATA_PATH, PROCESSED_TRAIN_PATH

def merge_data():
    all_dfs = []
    for file in TRAIN_DATA_PATH:
        df = pd.read_csv(file, sep=';')
        df["source_file"] = file.name
        df["fault_type"] = file.parent.name
        all_dfs.append(df)

    merged_df = pd.concat(all_dfs, ignore_index=True)
    return merged_df

def data_cleaning(merged_df):
    merged_df = merged_df.copy()
    merged_df = merged_df.drop(["anomaly", "changepoint"], axis=1)
    return merged_df

def save_processed_data(merged_df):
    PROCESSED_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(PROCESSED_TRAIN_PATH, index=False)

def main():
    merged_df = merge_data()
    merged_df = data_cleaning(merged_df)
    save_processed_data(merged_df)
    print("Preprocessing completed successfully")

if __name__ == "__main__":
    main()