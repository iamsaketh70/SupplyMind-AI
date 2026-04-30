"""Detects sudden positive-to-negative sentiment shifts for monitored entities."""

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_sentiment_history = defaultdict(list)
WINDOW_SIZE = 20
SHIFT_THRESHOLD = -0.4


def reset():
    """Clear all stored sentiment history."""
    _sentiment_history.clear()


def record_sentiment(entity, sentiment_score):
    """Append a sentiment score (-1 to 1) for the entity."""
    _sentiment_history[entity].append(sentiment_score)


def detect_shift(entity):
    """Check if entity experienced a sudden positive-to-negative sentiment shift."""
    history = _sentiment_history.get(entity, [])
    if len(history) < WINDOW_SIZE:
        return {"shift_detected": False, "delta": 0.0, "reason": "insufficient data"}

    recent = history[-WINDOW_SIZE // 2:]
    older = history[-WINDOW_SIZE: -WINDOW_SIZE // 2]

    recent_avg = sum(recent) / len(recent)
    older_avg = sum(older) / len(older)
    delta = recent_avg - older_avg

    shifted = delta < SHIFT_THRESHOLD and older_avg > 0
    return {
        "shift_detected": shifted,
        "delta": round(delta, 4),
        "older_avg": round(older_avg, 4),
        "recent_avg": round(recent_avg, 4),
        "reason": "positive→negative shift" if shifted else "stable",
    }


def risk_label_to_sentiment(label):
    """Convert a risk label string to a pseudo-sentiment score."""
    mapping = {
        "low risk": 0.5,
        "medium risk": 0.0,
        "high risk": -0.5,
        "critical risk": -1.0,
    }
    return mapping.get(label, 0.0)


if __name__ == "__main__":
    for i in range(10):
        record_sentiment("TestCorp", 0.6)
    for i in range(10):
        record_sentiment("TestCorp", -0.7)
    result = detect_shift("TestCorp")
    print(f"Shift result: {result}")
