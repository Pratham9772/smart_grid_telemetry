"""Central config for smart-grid telemetry pipeline."""
import os
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "smart_grid_telemetry.db")

VEHICLE_LOGS_TABLE = "vehicle_logs"
GPS_TELEMETRY_TABLE = "gps_telemetry"

# Battery thresholds (Li-ion pack, 400V nominal)
BATTERY_VOLTAGE_MIN = 280.0
BATTERY_VOLTAGE_MAX = 420.0
BATTERY_VOLTAGE_CRITICAL_LOW = 260.0
BATTERY_VOLTAGE_CRITICAL_HIGH = 450.0

# Load thresholds (kW)
LOAD_MIN = 0.0
LOAD_MAX = 350.0
LOAD_CRITICAL = 400.0

# Thermal thresholds (°C)
TEMP_MIN = -20.0
TEMP_MAX = 60.0
TEMP_CRITICAL = 80.0

# IQR outlier multiplier
IQR_MULTIPLIER = 1.5

# Isolation Forest params
IF_CONTAMINATION = 0.08
IF_N_ESTIMATORS = 150
IF_RANDOM_STATE = 42

# Logging
LOG_FILE = os.path.join(BASE_DIR, "pipeline.log")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = logging.INFO

# Synthetic data params
SAMPLE_SIZE = 1000
FLEET_IDS = [f"EV-{str(i).zfill(4)}" for i in range(1, 21)]

# Streamlit
STREAMLIT_PAGE_TITLE = "Smart-Grid Fleet Telemetry Dashboard"
STREAMLIT_LAYOUT = "wide"


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("smart_grid_pipeline")
    logger.setLevel(LOG_LEVEL)
    if not logger.handlers:
        fh = logging.FileHandler(LOG_FILE, mode="a")
        fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger
