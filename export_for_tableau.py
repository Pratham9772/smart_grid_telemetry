import sqlite3
import pandas as pd
from config import (
    DB_PATH, VEHICLE_LOGS_TABLE, GPS_TELEMETRY_TABLE,
    BATTERY_VOLTAGE_CRITICAL_LOW, BATTERY_VOLTAGE_CRITICAL_HIGH,
    LOAD_CRITICAL, TEMP_CRITICAL,
)

def export_for_tableau():
    conn = sqlite3.connect(DB_PATH)
    
    
    query = f"""
        SELECT
            vl.log_id, vl.vehicle_id, vl.battery_voltage,
            vl.current_load_kw, vl.temperature_c, vl.timestamp,
            gt.latitude, gt.longitude
        FROM {VEHICLE_LOGS_TABLE} vl
        INNER JOIN {GPS_TELEMETRY_TABLE} gt
            ON vl.vehicle_id = gt.vehicle_id AND vl.timestamp = gt.timestamp
        ORDER BY vl.timestamp DESC
    """
    df = pd.read_sql_query(query, conn)
    
    # Time features 
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = df['timestamp'].dt.date
    
    # threshold flags
    df['voltage_critical'] = (
        (df['battery_voltage'] < BATTERY_VOLTAGE_CRITICAL_LOW) |
        (df['battery_voltage'] > BATTERY_VOLTAGE_CRITICAL_HIGH)
    )
    df['load_critical'] = df['current_load_kw'] > LOAD_CRITICAL
    df['temp_critical'] = df['temperature_c'] > TEMP_CRITICAL
    
    # Any critical flag
    df['any_critical'] = df['voltage_critical'] | df['load_critical'] | df['temp_critical']
    
    
    # Save
    df.to_csv('tableau_data.csv', index=False)
    print(f"Exported {len(df)} rows to tableau_data.csv")
    print(f"Voltage critical: {df['voltage_critical'].sum()}")
    print(f"Load critical: {df['load_critical'].sum()}")
    print(f"Temp critical: {df['temp_critical'].sum()}")
    print(f"Any critical: {df['any_critical'].sum()}")
    
    conn.close()

if __name__ == "__main__":
    export_for_tableau()