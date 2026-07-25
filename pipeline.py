import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

from config import (
    DB_PATH, VEHICLE_LOGS_TABLE, GPS_TELEMETRY_TABLE,
    BATTERY_VOLTAGE_CRITICAL_LOW, BATTERY_VOLTAGE_CRITICAL_HIGH,
    LOAD_CRITICAL, TEMP_CRITICAL, IQR_MULTIPLIER,
    SAMPLE_SIZE, FLEET_IDS, setup_logging,
)

logger = setup_logging()


def init_database():
    """Drop and recreate SQLite schema with FK constraints."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute(f"DROP TABLE IF EXISTS {VEHICLE_LOGS_TABLE}")
        cursor.execute(f"DROP TABLE IF EXISTS {GPS_TELEMETRY_TABLE}")

        cursor.execute(f"""
            CREATE TABLE {GPS_TELEMETRY_TABLE} (
                gps_id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id TEXT NOT NULL,
                latitude REAL NOT NULL CHECK(latitude BETWEEN -90 AND 90),
                longitude REAL NOT NULL CHECK(longitude BETWEEN -180 AND 180),
                timestamp DATETIME NOT NULL
            )
        """)

        cursor.execute(f"""
            CREATE TABLE {VEHICLE_LOGS_TABLE} (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id TEXT NOT NULL,
                battery_voltage REAL ,
                current_load_kw REAL ,
                temperature_c REAL ,
                timestamp DATETIME )
        """)

        cursor.execute(f"CREATE INDEX idx_vehicle_id ON {VEHICLE_LOGS_TABLE}(vehicle_id)")
        conn.commit()
        logger.info("Database schema initialized")
    except sqlite3.Error as e:
        conn.rollback()
        logger.error(f"DB init failed: {e}")
        raise
    finally:
        conn.close()


def generate_synthetic_telemetry(n_records=SAMPLE_SIZE):
    """Generate mock fleet data with injected nulls and outliers."""
    base_time = datetime.now() - timedelta(days=30)
    data = {
        "vehicle_id": [random.choice(FLEET_IDS) for _ in range(n_records)],
        "battery_voltage": np.random.normal(350.0, 30.0, n_records).tolist(),
        "current_load_kw": np.random.normal(150.0, 50.0, n_records).tolist(),
        "temperature_c": np.random.normal(35.0, 15.0, n_records).tolist(),
        "latitude": np.random.normal(28.6, 0.5, n_records).tolist(),
        "longitude": np.random.normal(77.2, 0.5, n_records).tolist(),
        "timestamp": [base_time + timedelta(hours=i) for i in range(n_records)],
    }

    # Inject ~3% nulls in voltage
    null_idx = np.random.choice(n_records, size=int(n_records * 0.03), replace=False)
    for i in null_idx:
        data["battery_voltage"][i] = np.nan

    # Inject ~2% load outliers
    out_idx = np.random.choice(n_records, size=int(n_records * 0.02), replace=False)
    for i in out_idx:
        data["current_load_kw"][i] = np.random.choice([
            np.random.uniform(500, 800), np.random.uniform(-50, -10)
        ])

    df = pd.DataFrame(data)
    logger.info(f"Generated {len(df)} records")
    return df


def load_raw_to_database(df: pd.DataFrame):
    """Load raw df into normalized SQLite tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys = ON")
        for _, row in df.iterrows():
            cursor.execute(f"""
                INSERT INTO {GPS_TELEMETRY_TABLE} (vehicle_id, latitude, longitude, timestamp)
                VALUES (?, ?, ?, ?)
            """, (row["vehicle_id"], row["latitude"], row["longitude"], str(row["timestamp"])))

            cursor.execute(f"""
                INSERT INTO {VEHICLE_LOGS_TABLE} (vehicle_id, battery_voltage, current_load_kw, temperature_c, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (row["vehicle_id"], row["battery_voltage"], row["current_load_kw"],
                  row["temperature_c"], str(row["timestamp"])))
        conn.commit()
        logger.info(f"Loaded {len(df)} records into DB")
    except sqlite3.Error as e:
        conn.rollback()
        logger.error(f"Load failed: {e}")
        raise
    finally:
        conn.close()


def extract_joined_data() -> pd.DataFrame:
    """Extract unified telemetry via SQL INNER JOIN."""
    conn = sqlite3.connect(DB_PATH)
    try:
        query = f"""
            SELECT
                vl.log_id, vl.vehicle_id, vl.battery_voltage,
                vl.current_load_kw, vl.temperature_c, vl.timestamp AS log_timestamp,
                gt.latitude, gt.longitude, gt.timestamp AS gps_timestamp
            FROM {VEHICLE_LOGS_TABLE} vl
            INNER JOIN {GPS_TELEMETRY_TABLE} gt
                ON vl.vehicle_id = gt.vehicle_id AND vl.timestamp = gt.timestamp
            ORDER BY vl.timestamp DESC
        """
        df = pd.read_sql_query(query, conn)
        logger.info(f"Extracted {len(df)} joined records")
        return df
    except sqlite3.Error as e:
        logger.error(f"Extract failed: {e}")
        raise
    finally:
        conn.close()


def clean_null_values(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then backward-fill missing values."""
    df = df.copy()
    before = df.isnull().sum().sum()
    df = df.ffill().bfill()
    after = df.isnull().sum().sum()
    logger.info(f"Nulls: {before} -> {after}")
    return df


def remove_iqr_outliers(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Filter rows outside Q1 - 1.5*IQR and Q3 + 1.5*IQR."""
    df_clean = df.copy()
    rows_before = len(df_clean)
    for col in columns:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - IQR_MULTIPLIER * iqr
        upper = q3 + IQR_MULTIPLIER * iqr
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    logger.info(f"IQR filter: {rows_before} -> {len(df_clean)} rows")
    return df_clean.reset_index(drop=True)


def apply_engineering_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean flags for critical voltage/load/temp violations."""
    df = df.copy()
    df["voltage_critical"] = (
        (df["battery_voltage"] < BATTERY_VOLTAGE_CRITICAL_LOW) |
        (df["battery_voltage"] > BATTERY_VOLTAGE_CRITICAL_HIGH)
    )
    df["load_critical"] = df["current_load_kw"] > LOAD_CRITICAL
    df["temp_critical"] = df["temperature_c"] > TEMP_CRITICAL
    return df


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run full transform: clean nulls, IQR outliers, threshold flags, sort."""
    df = clean_null_values(df)
    df = remove_iqr_outliers(df, ["battery_voltage", "current_load_kw", "temperature_c"])
    df = apply_engineering_thresholds(df)
    df = df.sort_values("log_timestamp").reset_index(drop=True)
    return df


def get_aggregations() -> dict:
    """Compute per-vehicle SQL aggregations."""
    conn = sqlite3.connect(DB_PATH)
    try:
        aggregations = {
            "avg_voltage_by_vehicle": pd.read_sql_query(
                f"SELECT vehicle_id, AVG(battery_voltage) as avg_voltage FROM {VEHICLE_LOGS_TABLE} GROUP BY vehicle_id ORDER BY avg_voltage DESC",
                conn
            ),
            "max_load_by_vehicle": pd.read_sql_query(
                f"SELECT vehicle_id, MAX(current_load_kw) as max_load FROM {VEHICLE_LOGS_TABLE} GROUP BY vehicle_id ORDER BY max_load DESC",
                conn
            ),
            "count_stats": pd.read_sql_query(
                f"SELECT COUNT(*) as total_records, COUNT(DISTINCT vehicle_id) as fleet_size FROM {VEHICLE_LOGS_TABLE}",
                conn
            ),
            "avg_temp_by_vehicle": pd.read_sql_query(
                f"SELECT vehicle_id, AVG(temperature_c) as avg_temp FROM {VEHICLE_LOGS_TABLE} GROUP BY vehicle_id",
                conn
            ),
        }
        return aggregations
    finally:
        conn.close()


def run_full_pipeline():
    """End-to-end ETL: init -> generate -> load -> extract -> transform -> aggregate."""
    logger.info("Starting ETL pipeline")
    init_database()
    raw = generate_synthetic_telemetry()
    load_raw_to_database(raw)
    extracted = extract_joined_data()
    transformed = transform_data(extracted)
    aggregations = get_aggregations()
    logger.info("ETL pipeline complete")
    return transformed, aggregations


if __name__ == "__main__":
    df, aggs = run_full_pipeline()
    print(f"Records: {len(df)}, Fleet: {aggs['count_stats']['fleet_size'].values[0]}")
