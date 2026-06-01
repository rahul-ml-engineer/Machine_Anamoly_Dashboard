import pandas as pd

from config import Anamoly_RESULTS_PATH, STREAMING_RESULTS_PATH

def load_results():
    anomaly_df = pd.read_csv(Anamoly_RESULTS_PATH)
    streaming_df = pd.read_csv(STREAMING_RESULTS_PATH)
    return anomaly_df, streaming_df

def main():
    anomaly_df, streaming_df = load_results()
    print(anomaly_df["anomaly"].value_counts())
    print(streaming_df["prediction"].value_counts())


if __name__ == "__main__":
    main()  