import time
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMING_DIR = PROJECT_ROOT / "Data" / "streaming"
PROCESSED_DIR = PROJECT_ROOT / "Data" / "Processed"

SENSOR_COLUMNS = ["Temperature", "Pressure", "Current", "Accelerometer1RMS", "Volume Flow RateRMS"]

DEFAULT_REASON = "Machine operating within normal range."
DEFAULT_ACTION = "Continue monitoring."

PARAMETER_GUIDANCE = {
    "Temperature": {
        "risk": "Overheating",
        "action": "Check cooling, lubrication, and thermal load.",
    },
    "Pressure": {
        "risk": "Pressure instability",
        "action": "Inspect pump pressure, leaks, and blocked lines.",
    },
    "Current": {
        "risk": "Motor overload",
        "action": "Check motor load, wiring, and supply current.",
    },
    "Accelerometer1RMS": {
        "risk": "Vibration or rotor imbalance",
        "action": "Inspect rotor alignment, bearings, and mounting.",
    },
    "Volume Flow RateRMS": {
        "risk": "Flow restriction",
        "action": "Inspect valve position, filters, and pipe blockage.",
    },
}


st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    layout="wide",
)


@st.cache_data
def load_stream_data():
    stream_df = pd.read_csv(STREAMING_DIR / "streaming_results.csv")
    sensor_df = pd.read_csv(PROCESSED_DIR / "anomaly_results.csv", usecols=SENSOR_COLUMNS,).reset_index()
    alerts_df = pd.read_csv(STREAMING_DIR / "streaming_results_explained.csv")[
        [
            "index",
            "possible_reason",
            "severity",
            "recommended_action",
        ]
    ]

    stream_df = stream_df.merge(sensor_df, on="index", how="left")
    stream_df = stream_df.merge(alerts_df, on="index", how="left")
    stream_df["possible_reason"] = stream_df["possible_reason"].fillna(DEFAULT_REASON)
    stream_df["severity"] = stream_df["severity"].fillna("OK")
    stream_df["recommended_action"] = stream_df["recommended_action"].fillna(
        DEFAULT_ACTION
    )

    return stream_df


@st.cache_data
def load_baseline(sensor_columns):
    processed_df = pd.read_csv(PROCESSED_DIR / "anomaly_results.csv")

    normal_df = processed_df
    if "anomaly_prediction" in processed_df.columns:
        normal_df = processed_df[processed_df["anomaly_prediction"] == 1]

    baseline = normal_df[list(sensor_columns)].agg(["mean", "std"]).T
    baseline["std"] = baseline["std"].replace(0, pd.NA).fillna(1)

    return baseline


def get_parameter_drivers(row, baseline):
    drivers = []

    for parameter in SENSOR_COLUMNS:
        value = row.get(parameter)
        normal_mean = baseline.loc[parameter, "mean"]
        normal_std = baseline.loc[parameter, "std"]
        z_score = (value - normal_mean) / normal_std
        impact_score = abs(z_score)
        direction = "High" if z_score > 0 else "Low"
        guidance = PARAMETER_GUIDANCE[parameter]

        drivers.append(
            {
                "Parameter": parameter,
                "Current Value": round(value, 4),
                "Normal Mean": round(normal_mean, 4),
                "Deviation": round(z_score, 2),
                "Direction": direction,
                "Impact Score": round(impact_score, 2),
                "Likely Failure Cause": guidance["risk"],
                "Recommended Check": guidance["action"],
            }
        )

    return pd.DataFrame(drivers).sort_values("Impact Score", ascending=False)


def get_problem_sensors(row, baseline, min_impact=1.0):
    if row["status"] != "Anomaly":
        return "None"

    drivers_df = get_parameter_drivers(row, baseline)
    problem_df = drivers_df[drivers_df["Impact Score"] >= min_impact]

    if problem_df.empty:
        problem_df = drivers_df.head(3)

    return ", ".join(problem_df["Parameter"].tolist())


def add_problem_sensor_column(dataframe, baseline):
    result_df = dataframe.copy()
    result_df["problem_sensors"] = result_df.apply(
        lambda row: get_problem_sensors(row, baseline),
        axis=1,
    )
    return result_df


def format_driver_summary(drivers_df):
    top_driver = drivers_df.iloc[0]

    return (
        f"{top_driver['Parameter']} is the strongest failure driver right now "
        f"({top_driver['Direction'].lower()} by {top_driver['Impact Score']} sigma). "
        f"Likely cause: {top_driver['Likely Failure Cause']}."
    )


stream_df = load_stream_data()
baseline_df = load_baseline(tuple(SENSOR_COLUMNS))
first_anomaly_position = int(stream_df.index[stream_df["status"] == "Anomaly"][0])

st.title("Predictive Maintenance System")
st.markdown("---")

with st.sidebar:
    page = st.radio(
        "Dashboard page",
        ["Live Stream", "Sensor Visualization"],
    )

    st.markdown("---")
    st.header("Stream Controls")
    view_mode = st.selectbox(
        "View mode",
        ["Around first anomaly", "Anomalies only", "All data"],
    )
    rows_to_stream = st.slider("Rows to stream", 50, 2000, 500, 50)
    update_delay = st.slider("Update delay", 0.0, 1.0, 0.05, 0.05)

    if view_mode == "Around first anomaly":
        default_start_row = max(first_anomaly_position - 25, 0)
    else:
        default_start_row = 0

    start_row = st.number_input(
        "Start row",
        min_value=0,
        max_value=max(len(stream_df) - 1, 0),
        value=default_start_row,
        step=1,
    )

    st.caption(f"First anomaly appears at stream row {first_anomaly_position}.")

if view_mode == "Anomalies only":
    df = stream_df[stream_df["status"] == "Anomaly"].head(rows_to_stream)
else:
    df = stream_df.iloc[start_row : start_row + rows_to_stream]

stream_key = f"{view_mode}-{start_row}-{rows_to_stream}"
if st.session_state.get("stream_key") != stream_key:
    st.session_state.stream_key = stream_key
    st.session_state.stream_position = 0
    st.session_state.paused_on_anomaly = False

with st.sidebar:
    if st.button("Reset stream"):
        st.session_state.stream_position = 0
        st.session_state.paused_on_anomaly = False
        st.rerun()

status_placeholder = st.empty()
driver_placeholder = st.empty()
table_placeholder = st.empty()
control_placeholder = st.empty()

if df.empty:
    st.info("No stream rows available for the selected controls.")
    st.stop()

position = min(st.session_state.stream_position, len(df) - 1)
row = df.iloc[position]
live_df = df.iloc[: position + 1]
parameter_drivers_df = get_parameter_drivers(row, baseline_df)
top_parameters_df = parameter_drivers_df.head(5)
problem_sensors = get_problem_sensors(row, baseline_df)
live_df = add_problem_sensor_column(live_df, baseline_df)

if page == "Live Stream":
    with status_placeholder.container():
        st.subheader("Current Machine Status")

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Status", row["status"])
        col2.metric("Severity", row["severity"])
        col3.metric("Anomaly Score", round(row["anomaly_score"], 4))
        col4.metric("Prediction", row["prediction"])
        col5.metric("Stream Row", int(row["index"]))

        if row["status"] == "Anomaly":
            st.warning(row["possible_reason"])
            st.info(row["recommended_action"])
        else:
            st.success(row["possible_reason"])

    with driver_placeholder.container():
        st.subheader("Failure Driver Parameters")

        if row["status"] == "Anomaly":
            st.error(f"Problem sensors now: {problem_sensors}")

            if row["severity"] == "CRITICAL":
                st.error(format_driver_summary(parameter_drivers_df))
            else:
                st.warning(format_driver_summary(parameter_drivers_df))

            impact_chart_df = top_parameters_df.set_index("Parameter")[["Impact Score"]]
            st.bar_chart(impact_chart_df)

            st.dataframe(
                top_parameters_df[
                    [
                        "Parameter",
                        "Current Value",
                        "Normal Mean",
                        "Direction",
                        "Deviation",
                        "Likely Failure Cause",
                        "Recommended Check",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No failure driver detected for the current machine state.")
            st.dataframe(
                top_parameters_df[
                    [
                        "Parameter",
                        "Current Value",
                        "Normal Mean",
                        "Direction",
                        "Deviation",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )


    with table_placeholder.container():
        st.subheader("Live Alert Feed")

        display_columns = [
            "status",
            "severity",
            "anomaly_score",
            "problem_sensors",
            *SENSOR_COLUMNS,
            "possible_reason",
            "recommended_action",
        ]

        st.dataframe(
            live_df[display_columns].tail(10),
            use_container_width=True,
            hide_index=True,
        )

else:
    st.subheader("Current Machine Status")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Status", row["status"])
    col2.metric("Severity", row["severity"])
    col3.metric("Anomaly Score", round(row["anomaly_score"], 4))
    col4.metric("Prediction", row["prediction"])
    col5.metric("Stream Row", int(row["index"]))

    if row["status"] == "Anomaly":
        st.warning(f"Problem sensors now: {problem_sensors}")
    else:
        st.success("No problem sensors detected in the current row.")

    st.markdown("---")
    st.subheader("Live Sensor Values")

    sensor_cols = st.columns(len(SENSOR_COLUMNS))
    for sensor_col, sensor_name in zip(sensor_cols, SENSOR_COLUMNS):
        current_value = row[sensor_name]
        normal_mean = baseline_df.loc[sensor_name, "mean"]
        delta_value = current_value - normal_mean
        sensor_col.metric(
            sensor_name,
            round(current_value, 4),
            delta=round(delta_value, 4),
        )

    st.markdown("---")
    st.subheader("Sensor Live Visualization")

    selected_sensors = st.multiselect(
        "Sensors",
        SENSOR_COLUMNS,
        default=SENSOR_COLUMNS,
    )

    if selected_sensors:
        st.line_chart(live_df[selected_sensors], use_container_width=True)
    else:
        st.info("Select at least one sensor to show the live trend.")

    st.markdown("---")
    st.subheader("Problem Sensor Ranking")

    st.bar_chart(top_parameters_df.set_index("Parameter")[["Impact Score"]])

    st.dataframe(
        top_parameters_df[
            [
                "Parameter",
                "Current Value",
                "Normal Mean",
                "Direction",
                "Deviation",
                "Impact Score",
                "Likely Failure Cause",
                "Recommended Check",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.subheader("Sensor Alert History")

    sensor_history_columns = [
        "status",
        "severity",
        "anomaly_score",
        "problem_sensors",
        *SENSOR_COLUMNS,
    ]

    st.dataframe(
        live_df[sensor_history_columns].tail(20),
        use_container_width=True,
        hide_index=True,
    )

if row["status"] == "Anomaly":
    st.session_state.paused_on_anomaly = True

with control_placeholder.container():
    if st.session_state.paused_on_anomaly:
        st.warning("Stream paused because an anomaly was detected.")

        if st.button("Continue stream"):
            st.session_state.paused_on_anomaly = False
            st.session_state.stream_position = min(position + 1, len(df) - 1)
            st.rerun()
    elif position >= len(df) - 1:
        st.success("End of selected stream.")
    else:
        time.sleep(update_delay)
        st.session_state.stream_position = position + 1
        st.rerun()
