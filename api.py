from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

from pipeline import run_full_pipeline, extract_joined_data, transform_data, get_aggregations
from anomaly_detector import AnomalyDetector
from config import setup_logging, PROJECT_NAME, VERSION

logger = setup_logging()
app = FastAPI(title="Smart-Grid Telemetry API", description="ETL + ML anomaly detection REST API", version=VERSION)

latest_anomaly_results = None


class PipelineResponse(BaseModel):
    status: str
    records_processed: int
    fleet_size: int
    execution_time_ms: float


class TelemetryRecord(BaseModel):
    vehicle_id: str
    battery_voltage: float
    current_load_kw: float
    temperature_c: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timestamp: datetime
    voltage_critical: Optional[bool] = None
    load_critical: Optional[bool] = None
    temp_critical: Optional[bool] = None


class AnomalyRequest(BaseModel):
    features: List[str] = Field(default=["battery_voltage", "current_load_kw"])
    contamination: float = Field(default=0.08, ge=0.01, le=0.5)
    n_estimators: int = Field(default=150, ge=50, le=500)


class AnomalyResponse(BaseModel):
    status: str
    total_records: int
    anomalies_detected: int
    anomaly_percentage: float
    feature_importance: Dict[str, float]


@app.get("/health")
def health_check():
    logger.info("Health check")
    return {"status": "healthy", "project": PROJECT_NAME, "version": VERSION, "timestamp": datetime.now().isoformat()}


@app.post("/pipeline/run", response_model=PipelineResponse)
def run_pipeline():
    logger.info("Pipeline run requested")
    start = datetime.now()
    try:
        transformed, aggs = run_full_pipeline()
        elapsed = (datetime.now() - start).total_seconds() * 1000
        fleet = int(aggs["count_stats"]["fleet_size"].values[0])
        return PipelineResponse(status="success", records_processed=len(transformed), fleet_size=fleet, execution_time_ms=round(elapsed, 2))
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/telemetry", response_model=List[TelemetryRecord])
def get_telemetry(vehicle_id: Optional[str] = None, limit: int = 100):
    logger.info(f"Telemetry query: vehicle={vehicle_id}, limit={limit}")
    try:
        df = extract_joined_data()
        df = transform_data(df)
        if vehicle_id:
            df = df[df["vehicle_id"] == vehicle_id]
            if len(df) == 0:
                raise HTTPException(status_code=404, detail=f"No data for {vehicle_id}")
        df = df.head(limit)
        records = []
        for _, row in df.iterrows():
            records.append(TelemetryRecord(
                vehicle_id=row["vehicle_id"], battery_voltage=row["battery_voltage"],
                current_load_kw=row["current_load_kw"], temperature_c=row["temperature_c"],
                latitude=row.get("latitude"), longitude=row.get("longitude"),
                timestamp=row["log_timestamp"], voltage_critical=row.get("voltage_critical"),
                load_critical=row.get("load_critical"), temp_critical=row.get("temp_critical"),
            ))
        return records
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/telemetry/{vehicle_id}", response_model=List[TelemetryRecord])
def get_vehicle_telemetry(vehicle_id: str):
    return get_telemetry(vehicle_id=vehicle_id, limit=1000)


@app.get("/aggregations")
def get_aggregations_endpoint():
    logger.info("Aggregations requested")
    try:
        return {k: v.to_dict("records") for k, v in get_aggregations().items()}
    except Exception as e:
        logger.error(f"Aggregation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/anomaly/detect", response_model=AnomalyResponse)
def detect_anomalies(request: AnomalyRequest):
    logger.info(f"Anomaly detection: features={request.features}")
    try:
        df = extract_joined_data()
        df = transform_data(df)
        missing = set(request.features) - set(df.columns)
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing: {missing}")

        detector = AnomalyDetector(contamination=request.contamination, n_estimators=request.n_estimators, random_state=42)
        result = detector.detect_anomalies(df, request.features)
        global latest_anomaly_results
        latest_anomaly_results = result

        n_anom = int(result["is_anomaly"].sum())
        n_total = len(result)
        return AnomalyResponse(
            status="success", total_records=n_total, anomalies_detected=n_anom,
            anomaly_percentage=round((n_anom / n_total) * 100, 2) if n_total > 0 else 0.0,
            feature_importance=detector.get_feature_importance(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anomaly/results")
def get_anomaly_results():
    if latest_anomaly_results is None:
        raise HTTPException(status_code=404, detail="Run /anomaly/detect first")
    df = latest_anomaly_results
    anom = df[df["is_anomaly"]]
    return {
        "total_records": len(df), "anomalies_detected": int(df["is_anomaly"].sum()),
        "anomaly_percentage": round((df["is_anomaly"].sum() / len(df)) * 100, 2),
        "flagged_records": anom[["vehicle_id", "battery_voltage", "current_load_kw", "anomaly_score"]].to_dict("records"),
    }
