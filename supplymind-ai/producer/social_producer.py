"""Streams all 3 Reddit CSV files into the social_stream Kafka topic."""

import os
import sys
import json
import time
import pandas as pd
from kafka import KafkaProducer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    KAFKA_BROKER, TOPICS, STREAM_DELAY, DATA_DIR,
    SOCIAL_FILES, TEXT_COLUMN_KEYWORDS,
)

WALLSTREETBETS_ROW_LIMIT = 50_000
SKIP_SELFTEXT = {"[deleted]", "[removed]", "nan", ""}


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


def stream_file(producer, filepath, topic, source_label, max_rows=None):
    """Read a Reddit CSV and stream each row as JSON into the Kafka topic."""
    if not os.path.exists(filepath):
        print(f"[SKIP] File not found: {filepath}")
        return 0

    print(f"[SOCIAL] Loading {os.path.basename(filepath)} ...")
    nrows = max_rows if max_rows else None
    df = pd.read_csv(
        filepath,
        nrows=nrows,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="ignore",
    )

    has_title = any("title" in c.lower() for c in df.columns)
    has_selftext = any("selftext" in c.lower() for c in df.columns)
    title_col = next((c for c in df.columns if "title" in c.lower()), None)
    selftext_col = next((c for c in df.columns if "selftext" in c.lower()), None)

    if has_title and has_selftext:
        text_mode = "reddit_combined"
        print(f"[SOCIAL] Detected Reddit format: combining '{title_col}' + '{selftext_col}'")
    else:
        text_col = detect_text_column(df)
        if text_col is None:
            print(f"[SKIP] No text column found in {os.path.basename(filepath)}. Columns: {list(df.columns)}")
            return 0
        text_mode = "single"
        print(f"[SOCIAL] Detected text column: '{text_col}'")

    row_count = len(df)
    if max_rows:
        print(f"[SOCIAL] Limited to {max_rows} rows (loaded {row_count})")
    print(f"[SOCIAL] Streaming {row_count} rows ...")

    sent = 0
    for idx, row in df.iterrows():
        if text_mode == "reddit_combined":
            title = str(row.get(title_col, "")).strip()
            selftext = str(row.get(selftext_col, "")).strip()
            if selftext.lower() in SKIP_SELFTEXT:
                selftext = ""
            text = f"{title} {selftext}".strip()
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
            if col not in (title_col, selftext_col, text_col if text_mode == "single" else "") and col not in message:
                val = row[col]
                if pd.notna(val):
                    message[col] = str(val)[:200]
        producer.send(topic, value=message)
        sent += 1
        if sent % 1000 == 0:
            print(f"  [SOCIAL] Sent {sent} rows from {os.path.basename(filepath)}")
        time.sleep(STREAM_DELAY)

    producer.flush()
    print(f"[SOCIAL] Finished {os.path.basename(filepath)} — {sent} messages sent.")
    return sent


def main():
    """Entry point: create Kafka producer and stream all social files."""
    print("=" * 60)
    print("  SupplyMind AI — Social Producer")
    print("=" * 60)
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        print(f"[SOCIAL] Connected to Kafka at {KAFKA_BROKER}")
    except Exception as e:
        print(f"[ERROR] Could not connect to Kafka: {e}")
        return

    topic = TOPICS["social"]
    total = 0
    for fname in SOCIAL_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        max_rows = WALLSTREETBETS_ROW_LIMIT if "wallstreet_bets" in fname else None
        total += stream_file(producer, fpath, topic, source_label="reddit", max_rows=max_rows)

    producer.close()
    print(f"[SOCIAL] All done — {total} total messages sent to '{topic}'.")


if __name__ == "__main__":
    main()
