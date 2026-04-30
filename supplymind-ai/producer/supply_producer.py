"""Streams both supply chain CSV files into the supply_stream Kafka topic."""

import os
import sys
import json
import time
import pandas as pd
from kafka import KafkaProducer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    KAFKA_BROKER, TOPICS, STREAM_DELAY, DATA_DIR,
    SUPPLY_FILES, TEXT_COLUMN_KEYWORDS,
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

    print(f"[SUPPLY] Loading {os.path.basename(filepath)} ...")
    df = pd.read_csv(
        filepath,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="ignore",
    )
    text_col = detect_text_column(df)

    # global_supply_chain_risk_2026.csv has no real text — synthesize from metadata
    synthesize = False
    if text_col is None or (text_col and df[text_col].dropna().nunique() <= 3):
        if "Origin_Port" in df.columns and "Destination_Port" in df.columns:
            synthesize = True
            print(f"[SUPPLY] Synthesizing text from metadata columns in {os.path.basename(filepath)}")
        else:
            print(f"[SKIP] No text column found in {os.path.basename(filepath)}. Columns: {list(df.columns)}")
            return 0

    if not synthesize:
        print(f"[SUPPLY] Detected text column: '{text_col}' — streaming {len(df)} rows ...")

    sent = 0
    for idx, row in df.iterrows():
        if synthesize:
            parts = []
            for c in ["Product_Category", "Origin_Port", "Destination_Port",
                       "Transport_Mode", "Weather_Condition"]:
                if c in df.columns and pd.notna(row.get(c)):
                    parts.append(str(row[c]))
            risk_score = row.get("Geopolitical_Risk_Score", "")
            disrupted = row.get("Disruption_Occurred", 0)
            status = "disrupted" if disrupted == 1 else "on-track"
            text = f"Shipment {status}: {' / '.join(parts)}. Risk score: {risk_score}"
        else:
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
            print(f"  [SUPPLY] Sent {sent} rows from {os.path.basename(filepath)}")
        time.sleep(STREAM_DELAY)

    producer.flush()
    print(f"[SUPPLY] Finished {os.path.basename(filepath)} — {sent} messages sent.")
    return sent


def main():
    """Entry point: create Kafka producer and stream all supply chain files."""
    print("=" * 60)
    print("  SupplyMind AI — Supply Chain Producer")
    print("=" * 60)
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        print(f"[SUPPLY] Connected to Kafka at {KAFKA_BROKER}")
    except Exception as e:
        print(f"[ERROR] Could not connect to Kafka: {e}")
        return

    topic = TOPICS["supply"]
    total = 0
    for fname in SUPPLY_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        total += stream_file(producer, fpath, topic, source_label="supply_chain")

    producer.close()
    print(f"[SUPPLY] All done — {total} total messages sent to '{topic}'.")


if __name__ == "__main__":
    main()
