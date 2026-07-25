"""Isolation Forest wrapper for fleet anomaly detection."""
import numpy as np
import pandas as pd
from typing import Union, List
from sklearn.ensemble import IsolationForest
from config import IF_CONTAMINATION, IF_N_ESTIMATORS, IF_RANDOM_STATE, setup_logging

logger = setup_logging()


class AnomalyDetector:
    """Multi-variable anomaly detection using Isolation Forest."""

    def __init__(self, contamination=IF_CONTAMINATION, n_estimators=IF_N_ESTIMATORS, random_state=IF_RANDOM_STATE):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            bootstrap=False,
            n_jobs=-1,
        )
        self.feature_columns: List[str] = []
        self.is_fitted = False
        logger.info(f"Detector init: contamination={contamination}, n_estimators={n_estimators}")

    def fit(self, X: Union[pd.DataFrame, np.ndarray], feature_names: List[str] = None):
        """Train on feature matrix."""
        if isinstance(X, pd.DataFrame):
            self.feature_columns = list(X.columns)
            X_array = X.values
        else:
            X_array = np.asarray(X)
            self.feature_columns = feature_names or [f"f{i}" for i in range(X_array.shape[1])]

        if X_array.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X_array.shape}")

        logger.info(f"Training on {X_array.shape[0]} samples, {X_array.shape[1]} features")
        self.model.fit(X_array)
        self.is_fitted = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict: -1 = anomaly, +1 = normal."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        X_array = X[self.feature_columns].values if isinstance(X, pd.DataFrame) else np.asarray(X)
        labels = self.model.predict(X_array)
        logger.info(f"Predicted: {np.sum(labels == -1)} anomalies, {np.sum(labels == 1)} normal")
        return labels

    def decision_function(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Anomaly scores: negative = outlier."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        X_array = X[self.feature_columns].values if isinstance(X, pd.DataFrame) else np.asarray(X)
        scores = self.model.decision_function(X_array)
        logger.info(f"Scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
        return scores

    def detect_anomalies(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Fit, predict, score, and annotate df with anomaly flags."""
        missing = set(feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        X = df[feature_cols].copy()
        self.fit(X)
        labels = self.predict(X)
        scores = self.decision_function(X)

        result = df.copy()
        result["anomaly_label"] = labels
        result["is_anomaly"] = (labels == -1)
        result["anomaly_score"] = scores

        n_anom = result["is_anomaly"].sum()
        logger.info(f"Detected {n_anom}/{len(result)} anomalies ({n_anom/len(result)*100:.1f}%)")
        return result

    def get_feature_importance(self) -> dict:
        """Approximate importance by split frequency across all trees."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        counts = {col: 0 for col in self.feature_columns}
        for estimator in self.model.estimators_:
            tree = estimator.tree_
            for node in tree.feature:
                if node != -2:
                    counts[self.feature_columns[node]] += 1

        total = sum(counts.values())
        return {k: (v / total if total > 0 else 0.0) for k, v in counts.items()}


if __name__ == "__main__":
    np.random.seed(42)
    test = pd.DataFrame({
        "battery_voltage": np.concatenate([np.random.normal(350, 20, 95), np.random.normal(500, 10, 5)]),
        "current_load_kw": np.concatenate([np.random.normal(150, 40, 95), np.random.normal(600, 20, 5)]),
    })
    det = AnomalyDetector(contamination=0.05)
    res = det.detect_anomalies(test, ["battery_voltage", "current_load_kw"])
    print(f"Anomalies: {res['is_anomaly'].sum()}/{len(res)}")
