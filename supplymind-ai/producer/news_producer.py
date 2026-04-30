"""Streams all-data.csv and financial_news_events.csv into the news_stream Kafka topic."""

import os
import sys
import json
import time
import pandas as pd
from kafka import KafkaProducer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    KAFKA_BROKER, TOPICS, STREAM_DELAY, DATA_DIR,
    NEWS_FILES, TEXT_COLUMN_KEYWORDS,
)


def detect_text_column(df):
    """Return the first column that has a text keyword in its name and contains real strings."""
    for col in df.columns:
        col_lower = str(col).lower()
        for kw in TEXT_COLUMN_KEYWORDS:
            if kw in col_lower:
                sample = df[col].dropna().head(20)
                if sample.empty:
                    continue
                if sample.astype(str).str.len().mean() > 5:
                    return col
    return None


def stream_file(producer, filepath, topic, source_label):
    """Read a CSV and stream each row as JSON into the Kafka topic."""
    if not os.path.exists(filepath):
        print(f"[SKIP] File not found: {filepath}")
        return 0

    print(f"[NEWS] Loading {os.path.basename(filepath)} ...")
    df = pd.read_csv(
        filepath,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="ignore",
    )

    # all-data.csv has no header — col 0 is sentiment, col 1 is text
    if all(isinstance(c, int) or str(c).isdigit() for c in df.columns) or \
       (len(df.columns) == 2 and detect_text_column(df) is None):
        df = pd.read_csv(
            filepath, header=None, names=["sentiment", "text"],
            on_bad_lines="skip", encoding="utf-8", encoding_errors="ignore",
        )

    text_col = detect_text_column(df)
    if text_col is None:
        print(f"[SKIP] No text column found in {os.path.basename(filepath)}. Columns: {list(df.columns)}")
        return 0

    print(f"[NEWS] Detected text column: '{text_col}' — streaming {len(df)} rows ...")
    sent = 0
    for idx, row in df.iterrows():
        text = str(row.get(text_col, "")).strip()
        if not text or text.lower() in ("nan", ""):
            continue
        message = {
            "text": text,
            "source": source_label,
            "timestamp": time.time(),
            "file": os.path.basename(filepath),
        }
        for col in df.columns:
            if col != text_col and col not in message:
                val = row[col]
                if pd.notna(val):
                    message[col] = str(val)
        producer.send(topic, value=message)
        sent += 1
        if sent % 1000 == 0:
            print(f"  [NEWS] Sent {sent} rows from {os.path.basename(filepath)}")
        time.sleep(STREAM_DELAY)

    producer.flush()
    print(f"[NEWS] Finished {os.path.basename(filepath)} — {sent} messages sent.")
    return sent


def main():
    """Entry point: create Kafka producer and stream all news files."""
    print("=" * 60)
    print("  SupplyMind AI — News Producer")
    print("=" * 60)
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        print(f"[NEWS] Connected to Kafka at {KAFKA_BROKER}")
    except Exception as e:
        print(f"[ERROR] Could not connect to Kafka: {e}")
        return

    topic = TOPICS["news"]
    total = 0
    for fname in NEWS_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        total += stream_file(producer, fpath, topic, source_label="news")

    producer.close()
    print(f"[NEWS] All done — {total} total messages sent to '{topic}'.")


if __name__ == "__main__":
    main()
