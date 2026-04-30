"""7-day disruption probability forecast using GradientBoostingRegressor."""

import os
import sys
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _build_features(series):
    """Build a feature vector from a time series of daily risk scores."""
    arr = np.array(series, dtype=float)
    rolling_mean = float(np.mean(arr[-7:]))
    rolling_std = float(np.std(arr[-7:]))
    rolling_max = float(np.max(arr[-7:]))
    rolling_min = float(np.min(arr[-7:]))
    last_val = float(arr[-1])
    trend = float(arr[-1] - arr[0]) / max(len(arr), 1)
    high_risk_frac = float(np.sum(arr > 2.5)) / max(len(arr), 1)
    return [rolling_mean, rolling_std, rolling_max, rolling_min, last_val, trend, high_risk_frac]


def _risk_level_to_label(score):
    """Convert a numeric risk score to a human-readable label."""
    if score < 1.5:
        return "LOW"
    elif score < 2.5:
        return "MEDIUM"
    elif score < 3.5:
        return "HIGH"
    return "CRITICAL"


def forecast(daily_risk_scores, horizon=7):
    """Forecast the next `horizon` days of risk scores using GBM."""
    if len(daily_risk_scores) < 10:
        last = daily_risk_scores[-1] if daily_risk_scores else 2.0
        return [
            {"day": d + 1, "predicted_risk": round(float(last), 2), "label": _risk_level_to_label(last)}
            for d in range(horizon)
        ]

    arr = np.array(daily_risk_scores, dtype=float)
    X, y = [], []
    window = 7
    for i in range(window, len(arr)):
        feats = _build_features(arr[i - window: i])
        X.append(feats)
        y.append(arr[i])

    X = np.array(X)
    y = np.array(y)

    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X, y)

    predictions = []
    current_series = list(arr)
    for d in range(horizon):
        feats = np.array([_build_features(current_series[-window:])])
        pred = float(model.predict(feats)[0])
        pred = max(1.0, min(4.0, pred))
        predictions.append({
            "day": d + 1,
            "predicted_risk": round(pred, 2),
            "label": _risk_level_to_label(pred),
        })
        current_series.append(pred)

    return predictions


def forecast_multiple_entities(entity_history_dict, horizon=7):
    """Forecast for multiple entities and return a dict of entity → predictions."""
    results = {}
    for entity, scores in entity_history_dict.items():
        results[entity] = forecast(scores, horizon)
    return results


if __name__ == "__main__":
    np.random.seed(42)
    fake_history = list(np.random.uniform(1.0, 3.5, size=30))
    preds = forecast(fake_history)
    print("7-day forecast:")
    for p in preds:
        print(f"  Day {p['day']}: risk={p['predicted_risk']} ({p['label']})")
