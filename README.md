![Dashboard Preview](assets/dasboard.png)
# Smart-Grid Fleet Telemetry Pipeline

Local data engineering pipeline for electric fleet telemetry. Ingests synthetic vehicle data (battery voltage, load, GPS, temperature) into SQLite, runs IQR outlier filtering and Isolation Forest anomaly detection, and serves results via Streamlit dashboard and FastAPI.

## Stack

- Python 3.11, Pandas, NumPy
- SQLite3 (native, no external DB needed)
- Scikit-learn (Isolation Forest)
- Streamlit (dashboard)
- FastAPI + Uvicorn (REST API)
- pytest (unit tests)
- Docker + Docker Compose

## Quick Start

```bash
pip install -r requirements.txt
pytest test_pipeline.py -v        # run tests
python pipeline.py                 # run ETL standalone
streamlit run app.py             # launch dashboard
uvicorn api:app --reload         # start API
```

## Pipeline Flow

1. `generate_synthetic_telemetry()` — creates 1000 mock records with injected nulls/outliers
2. `load_raw_to_database()` — splits into `gps_telemetry` and `vehicle_logs` tables with FK constraint
3. `extract_joined_data()` — SQL INNER JOIN on `vehicle_id` + `timestamp`
4. `transform_data()` — ffill/bfill nulls, IQR outlier removal, engineering threshold flags
5. `AnomalyDetector.detect_anomalies()` — Isolation Forest on voltage + load
6. Streamlit renders KPIs, time-series charts, anomaly tables, CSV export

## DB Schema

```
gps_telemetry(gps_id PK, vehicle_id, latitude, longitude, timestamp)
vehicle_logs(log_id PK, vehicle_id FK, battery_voltage, current_load_kw, temperature_c, timestamp)
```

FK has `ON DELETE CASCADE`. `PRAGMA foreign_keys = ON` enforced.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service status |
| POST | `/pipeline/run` | Execute full ETL |
| GET | `/telemetry` | All records (optional `vehicle_id` filter) |
| GET | `/telemetry/{vehicle_id}` | Records for one vehicle |
| GET | `/aggregations` | SQL KPIs |
| POST | `/anomaly/detect` | Run ML detection |
| GET | `/anomaly/results` | Latest anomaly results |

## Thresholds

| Metric | Safe Range | Critical |
|--------|-----------|----------|
| Battery Voltage | 280–420 V | < 260 or > 450 V |
| Current Load | 0–350 kW | > 400 kW |
| Temperature | –20 to 60 °C | > 80 °C |
```

## Tests

5 test cases covering null handling, IQR math verification, threshold flags, DB schema validation, and empty DataFrame edge cases.

```bash
pytest test_pipeline.py -v
```
