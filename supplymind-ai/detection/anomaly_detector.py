"""Z-score spike detection on entity mention counts."""

import os
import sys
import uuid
import time
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ZSCORE_THRESHOLD = 3.0

_mention_history = defaultdict(list)


def reset():
    """Clear all stored mention history."""
    _mention_history.clear()


def update_count(entity, count=1):
    """Record a new mention count for an entity."""
    _mention_history[entity].append(count)


def detect_anomaly(entity, current_count):
    """Check if current_count is a z-score anomaly for the given entity."""
    history = _mention_history.get(entity, [])
    if len(history) < 5:
        return {
            "is_anomaly": False,
            "zscore": 0.0,
            "baseline_mean": current_count,
            "current_count": current_count,
            "multiplier": 1.0,
        }

    arr = np.array(history)
    mean = float(np.mean(arr))
    std = float(np.std(arr))

    if std == 0:
        zscore = 0.0
    else:
        zscore = (current_count - mean) / std

    multiplier = current_count / mean if mean > 0 else 1.0

    return {
        "is_anomaly": zscore > ZSCORE_THRESHOLD,
        "zscore": round(zscore, 3),
        "baseline_mean": round(mean, 2),
        "current_count": current_count,
        "multiplier": round(multiplier, 2),
    }


def create_alert(entity, risk_label, zscore, baseline_mean, current_count, sample_text=""):
    """Build a structured alert dict for an anomalous entity."""
    return {
        "alert_id": str(uuid.uuid4())[:8],
        "entity": entity,
        "risk_label": risk_label,
        "zscore": round(zscore, 3),
        "baseline_mean": round(baseline_mean, 2),
        "current_count": current_count,
        "multiplier": round(current_count / baseline_mean, 2) if baseline_mean > 0 else 1.0,
        "sample_text": str(sample_text)[:300],
        "timestamp": time.time(),
    }


def process_entity(entity, current_count, risk_label="unknown", sample_text=""):
    """Update history, detect anomaly, and return alert if triggered."""
    result = detect_anomaly(entity, current_count)
    update_count(entity, current_count)

    alert = None
    if result["is_anomaly"]:
        alert = create_alert(
            entity, risk_label, result["zscore"],
            result["baseline_mean"], current_count, sample_text,
        )
    return result, alert


if __name__ == "__main__":
    for i in range(20):
        count = 5 if i < 18 else 50
        res, alert = process_entity("TSMC", count, "high risk", "TSMC disruption news")
        if alert:
            print(f"ALERT: {alert}")
    print("Done — anomaly detection test complete.")
