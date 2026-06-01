import pandas as pd

from config import Anamoly_RESULTS_PATH, STREAMING_RESULTS_PATH, FINAL_STREAMING_RESULTS_PATH

def load_results():
    anomaly_df = pd.read_csv(Anamoly_RESULTS_PATH)
    streaming_df = pd.read_csv(STREAMING_RESULTS_PATH)
    return anomaly_df, streaming_df


def main():
    feature_df, stream_df = load_results()

    stream_df["Temperature"] = feature_df["Temperature"]
    stream_df["Pressure"] = feature_df["Pressure"]
    stream_df["Current"] = feature_df["Current"]
    stream_df["Accelerometer1RMS"] = (feature_df["Accelerometer1RMS"])
    stream_df["Volume Flow RateRMS"] = (feature_df["Volume Flow RateRMS"])

    reasons = []

    for _, row in stream_df.iterrows():
        reason = []

        if row["Accelerometer1RMS"] > 0.03:
            reason.append("Possible rotor imbalance")

        if row["Temperature"] > 70:
            reason.append("Possible overheating")

        if row["Pressure"] < 0.4:
            reason.append("Possible valve blockage")

        if row["Current"] > 3:
            reason.append("Possible motor overload")

        if row["Volume Flow RateRMS"] < 29:
            reason.append("Possible cavitation or low flow")

        if len(reason) == 0:
            reason.append("System operating normally")

        reasons.append(" | ".join(reason))

    stream_df["possible_reason"] = reasons

    severity = []

    for score in stream_df["anomaly_score"]:

        if score < -0.10:
            severity.append("CRITICAL")

        elif score < 0:
            severity.append("WARNING")

        else:
            severity.append("NORMAL")

    stream_df["severity"] = severity

    actions = []

    for reason in stream_df["possible_reason"]:

        if "rotor imbalance" in reason:
            actions.append(
                "Inspect rotor alignment and bearings"
            )

        elif "overheating" in reason:
            actions.append(
                "Inspect cooling system"
            )

        elif "valve blockage" in reason:
            actions.append(
                "Inspect inlet and outlet valves"
            )

        elif "motor overload" in reason:
            actions.append(
                "Check electrical load and motor health"
            )

        elif "cavitation" in reason:
            actions.append(
                "Check flow rate and pump inlet"
            )

        else:
            actions.append(
                "No maintenance needed"
            )

    stream_df["recommended_action"] = actions

    print(stream_df[["prediction", "anomaly_score", "severity", "possible_reason", "recommended_action"]].head())

    alerts_df = stream_df[stream_df["prediction"] == -1]
    alerts_df.to_csv(FINAL_STREAMING_RESULTS_PATH, index=False)
    print(f"\nTotal Alerts: {len(alerts_df)}")

if __name__ == "__main__":
    main()