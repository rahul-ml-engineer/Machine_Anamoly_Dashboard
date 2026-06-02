# Machine Anamoly Dashboard

A Streamlit dashboard for predictive maintenance and remaining useful life (RUL) style monitoring. The app streams machine sensor readings, highlights anomaly status, ranks likely failure-driver parameters, and recommends maintenance checks for abnormal machine behavior.

## Live Demo

Streamlit Cloud: **https://machineruldashboard-2ueifxsz2x7chrjqdcgf4y.streamlit.app/**

## Features

- Real-time machine status stream
- Anomaly severity and anomaly score display
- Sensor-level failure-driver ranking
- Recommended maintenance actions for detected issues
- Live sensor visualization with selectable parameters
- Alert feed and recent sensor history

## Project Structure

```text
Machine_RUL_Dashboard/
+-- Dashboard/
|   +-- app.py
+-- Data/
|   +-- Processed/
|   +-- raw/
|   +-- streaming/
+-- models/
+-- Notebook/
+-- src/
|   +-- config.py
|   +-- stream.py
+-- requirements.txt
+-- README.md
```

## Local Setup

1. Clone the repository.

```bash
git clone <your-repository-url>
cd Machine_RUL_Dashboard
```

2. Create and activate a virtual environment.

```bash
python -m venv venv
venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app.

```bash
streamlit run Dashboard/app.py
```

## Streamlit Cloud Deployment

Use these settings when deploying on Streamlit Cloud:

- **Repository:** your GitHub repository
- **Branch:** main
- **Main file path:** `Dashboard/app.py`
- **Python dependencies:** `requirements.txt`

Make sure the required data files are available in:

- `Data/Processed/anomaly_results.csv`
- `Data/streaming/streaming_results.csv`
- `Data/streaming/streaming_results_explained.csv`

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## License

This project is licensed under the terms of the included `LICENSE` file.
