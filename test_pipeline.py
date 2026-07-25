import pytest
import pandas as pd
import numpy as np
import sqlite3
from config import DB_PATH, VEHICLE_LOGS_TABLE, GPS_TELEMETRY_TABLE, IQR_MULTIPLIER, setup_logging
from pipeline import init_database, clean_null_values, remove_iqr_outliers, apply_engineering_thresholds, transform_data

logger = setup_logging()


@pytest.fixture
def mock_df():
    return pd.DataFrame({
        "vehicle_id": ["EV-0001"] * 10,
        "battery_voltage": [350.0, 352.0, np.nan, 348.0, 355.0, 351.0, 349.0, 500.0, 347.0, 353.0],
        "current_load_kw": [150.0, 155.0, 148.0, np.nan, 152.0, 600.0, 149.0, 151.0, 154.0, 147.0],
        "temperature_c": [35.0, 36.0, 34.0, 35.5, np.nan, 37.0, 85.0, 33.0, 35.2, 36.8],
        "log_timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
    })


@pytest.fixture
def clean_df():
    np.random.seed(42)
    volts = np.concatenate([np.random.normal(350, 10, 19), [500.0]])
    return pd.DataFrame({
        "vehicle_id": [f"EV-{i:04d}" for i in range(20)],
        "battery_voltage": volts,
        "current_load_kw": np.random.normal(150, 20, 20),
        "temperature_c": np.random.normal(35, 5, 20),
        "log_timestamp": pd.date_range("2024-01-01", periods=20, freq="h"),
    })


@pytest.fixture
def empty_df():
    return pd.DataFrame(columns=["vehicle_id", "battery_voltage", "current_load_kw", "temperature_c", "log_timestamp"])


def test_null_handling(mock_df):
    before = mock_df.isnull().sum().sum()
    assert before > 0
    cleaned = clean_null_values(mock_df)
    assert cleaned.isnull().sum().sum() == 0
    # ffill: index 2 should get 352.0 from index 1
    assert cleaned.loc[2, "battery_voltage"] == 352.0


def test_iqr_outliers(clean_df):
    rows_before = len(clean_df)
    cleaned = remove_iqr_outliers(clean_df, ["battery_voltage"])
    assert len(cleaned) < rows_before
    assert 500.0 not in cleaned["battery_voltage"].values

    # Verify math
    q1 = clean_df["battery_voltage"].quantile(0.25)
    q3 = clean_df["battery_voltage"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - IQR_MULTIPLIER * iqr
    upper = q3 + IQR_MULTIPLIER * iqr
    assert (cleaned["battery_voltage"] >= lower).all()
    assert (cleaned["battery_voltage"] <= upper).all()


def test_threshold_flags(mock_df):
    cleaned = clean_null_values(mock_df)
    flagged = apply_engineering_thresholds(cleaned)
    for col in ["voltage_critical", "load_critical", "temp_critical"]:
        assert col in flagged.columns

    # 500V should be flagged
    high_v = flagged[flagged["battery_voltage"] == 500.0].index
    if len(high_v) > 0:
        assert flagged.loc[high_v[0], "voltage_critical"] == True

    # 600 kW should be flagged
    high_l = flagged[flagged["current_load_kw"] == 600.0].index
    if len(high_l) > 0:
        assert flagged.loc[high_l[0], "load_critical"] == True

    # 85°C should be flagged
    high_t = flagged[flagged["temperature_c"] == 85.0].index
    if len(high_t) > 0:
        assert flagged.loc[high_t[0], "temp_critical"] == True


def test_database_schema():
    init_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:

        cursor.execute(f"PRAGMA table_info({GPS_TELEMETRY_TABLE})")
        gps_cols = {row[1] for row in cursor.fetchall()}
        assert {"gps_id", "vehicle_id", "latitude", "longitude", "timestamp"}.issubset(gps_cols)

        cursor.execute(f"PRAGMA table_info({VEHICLE_LOGS_TABLE})")
        vl_cols = {row[1] for row in cursor.fetchall()}
        assert {"log_id", "vehicle_id", "battery_voltage", "current_load_kw", "temperature_c", "timestamp"}.issubset(vl_cols)

        cursor.execute(f"PRAGMA index_list({VEHICLE_LOGS_TABLE})")
        indexes = {row[1] for row in cursor.fetchall()}
        assert "idx_vehicle_id" in indexes
    finally:
        conn.close()


def test_empty_df(empty_df):
    assert len(empty_df) == 0
    cleaned = clean_null_values(empty_df)
    assert len(cleaned) == 0
    flagged = apply_engineering_thresholds(cleaned)
    assert len(flagged) == 0
    assert "voltage_critical" in flagged.columns


def test_full_transform(mock_df):
    transformed = transform_data(mock_df)
    assert transformed.isnull().sum().sum() == 0
    assert "voltage_critical" in transformed.columns
    assert "load_critical" in transformed.columns
    assert "temp_critical" in transformed.columns
    timestamps = pd.to_datetime(transformed["log_timestamp"])
    assert timestamps.is_monotonic_increasing
