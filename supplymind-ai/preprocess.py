"""Pre-process all datasets: load, filter, score risk, extract entities with
BERT NER + keyword augmentation, and save to parquet/CSV."""

import os
import sys
import re
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config.settings import (
    DATA_DIR, NEWS_FILES, SOCIAL_FILES, SUPPLY_FILES,
    TEXT_COLUMN_KEYWORDS, KNOWN_COMPANIES, KNOWN_REGIONS,
)
from nlp.risk_scorer import score_batch

# ── Supply chain relevance keywords (strict domain vocabulary) ───────────────

SUPPLY_CHAIN_CORE = [
    "supply chain", "logistics", "procurement", "sourcing",
    "shipping", "freight", "cargo", "container", "vessel",
    "warehouse", "inventory", "distribution", "fulfillment",
    "manufactur", "factory", "assembly", "production line",
    "semiconductor", "chip shortage", "wafer", "foundry",
    "port", "port congestion", "shipping delay",
    "disruption", "disrupt", "bottleneck", "gridlock",
    "shortage", "scarcity", "stockout", "deficit",
    "tariff", "trade war", "sanction", "embargo",
    "supplier", "vendor", "tier-1", "tier-2",
    "raw material", "commodity", "crude oil", "natural gas",
    "lithium", "steel", "copper", "rare earth", "cobalt",
    "import", "export", "customs", "cross-border",
    "lead time", "backlog", "on-time delivery",
    "recall", "quality defect", "contamination",
    "trucking", "rail freight", "intermodal",
    "ev battery", "battery supply", "cathode",
    "pharma", "drug supply", "vaccine supply",
]

RISK_EVENT_TERMS = [
    "earthquake", "tsunami", "flood", "typhoon", "hurricane", "cyclone",
    "wildfire", "drought", "storm", "volcanic",
    "pandemic", "outbreak", "lockdown", "quarantine",
    "war", "conflict", "invasion", "military strike",
    "cyber attack", "ransomware", "data breach",
    "factory fire", "explosion", "plant closure", "production halt",
    "bankruptcy", "insolvency", "collapse",
    "port closure", "canal blocked", "blockade",
    "labor strike", "walkout", "protest",
    "price surge", "price spike", "inflation",
    "recession", "downturn", "economic crisis",
    "geopolit", "tension", "escalat",
    "power outage", "blackout", "energy crisis",
]

SOCIAL_NOISE_TERMS = [
    "yolo", "tendies", "tendie", "diamond hands", "paper hands",
    "to the moon", "rocket emoji", "apes together",
    "wife's boyfriend", "loss porn", "gain porn",
    "options play", "calls on", "puts on",
    "robinhood", "fidelity", "webull", "etrade",
    "what broker", "best platform", "how do i start",
    "personal finance", "budget", "savings account",
    "credit card", "mortgage rate", "student loan",
    "crypto", "bitcoin", "ethereum", "nft", "doge",
    "technical analysis", "chart pattern", "support level",
    "resistance level", "moving average", "rsi ",
    "wsb", "wallstreetbets", "degener",
]


def relevance_score(text):
    """Count how many supply-chain / risk-event terms appear."""
    t = text.lower()
    return (sum(1 for kw in SUPPLY_CHAIN_CORE if kw in t) +
            sum(1 for kw in RISK_EVENT_TERMS if kw in t))


def noise_score(text):
    """Count how many social-media noise terms appear."""
    t = text.lower()
    return sum(1 for kw in SOCIAL_NOISE_TERMS if kw in t)


def detect_text_column(df):
    """Auto-detect the best text column by keyword in name + avg string length."""
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


def extract_companies_keyword(text):
    """Fast company extraction using the expanded known-companies list."""
    t = text.lower()
    found = [c for c in KNOWN_COMPANIES if c.lower() in t]
    return found[:15]


def extract_locations_keyword(text):
    """Fast location extraction using the expanded known-regions list."""
    t = text.lower()
    found = [r for r in KNOWN_REGIONS if r.lower() in t]
    return found[:15]


def try_load_ner():
    """Attempt to load BERT NER pipeline; return None if unavailable."""
    try:
        from transformers import pipeline as hf_pipeline
        print("[NLP] Loading BERT NER model (dslim/bert-base-NER) ...")
        ner = hf_pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device=-1,
        )
        print("[NLP] BERT NER model loaded successfully.")
        return ner
    except Exception as e:
        print(f"[WARN] Could not load BERT NER: {e}")
        print("[INFO] Falling back to keyword-only entity extraction.")
        return None


NER_JUNK_ORG = {
    "the", "new", "all", "big", "top", "best", "free", "just",
    "get", "buy", "sell", "long", "short", "call", "put",
    "and", "for", "not", "but", "are", "was", "has",
    "usa", "gdp", "sec", "fed", "ipo", "ceo", "etf",
    "news", "finance", "market", "stock", "trading", "money",
    "air", "general", "national", "international", "global",
    "north", "south", "east", "west", "central", "united", "states",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "december",
    "real", "bill", "edit", "update",
    # Supply chain dataset structured fields misclassified as ORG
    "pharmaceuticals", "port congestion", "rail", "road", "sea",
    "electronics", "auto", "textiles", "machinery", "food",
    "chemicals", "metals", "plastics", "construction", "energy",
    "agriculture", "mining", "furniture", "paper", "beverages",
    "healthcare", "transportation", "utilities", "services",
    "consumer", "industrial", "technology", "defense",
    "disrupted", "on-track", "shipment",
    "clear", "rain", "fog", "storm", "snow", "hurricane", "typhoon",
    "per", "text", "nan",
    # Common NER false positives
    "inc", "corp", "ltd", "llc", "plc", "co", "pre", "post",
    "perishables", "ables", "non-perishables", "durables",
    "raw materials", "finished goods", "bulk",
    "q1", "q2", "q3", "q4", "fy", "yoy", "qoq",
    "api", "gpu", "cpu", "pdf",
}

NER_JUNK_LOC = NER_JUNK_ORG | {
    "sea", "ocean", "river", "lake", "mountain", "island",
    "road", "rail", "highway", "airport", "port",
    "hurricane", "typhoon", "storm", "rain", "fog", "clear",
    "snow", "wind", "hail", "tornado",
    "electronics", "auto", "pharmaceuticals", "machinery",
    "textiles", "chemicals", "metals", "food",
    "per", "text", "nan", "disrupted", "on-track",
    "earth", "world", "home", "online", "local",
    "perishables", "ables", "non-perishables", "durables",
    "marseille",
}


def extract_entities_ner(text, ner_pipeline):
    """Use BERT NER to extract ORG and LOC entities, augmented with keyword list.

    Applies strict junk filtering to remove false positives from structured
    supply chain data fields (weather, transport modes, product categories).
    """
    companies = set()
    locations = set()

    if ner_pipeline and text and len(text.strip()) > 5:
        try:
            entities = ner_pipeline(str(text)[:512])
            for ent in entities:
                label = ent.get("entity_group", "")
                word = ent.get("word", "").strip().replace("##", "")
                score = ent.get("score", 0.0)

                if len(word) < 3 or score < 0.75:
                    continue
                if not re.search(r"[a-zA-Z]{3,}", word):
                    continue

                word_lower = word.lower()

                if label == "ORG" and word_lower not in NER_JUNK_ORG:
                    companies.add(word)
                elif label in ("LOC", "GPE") and word_lower not in NER_JUNK_LOC:
                    locations.add(word)
        except Exception:
            pass

    t = text.lower() if text else ""
    for c in KNOWN_COMPANIES:
        if len(c) <= 2:
            if f" {c.lower()} " in f" {t} ":
                companies.add(c)
        elif c.lower() in t:
            companies.add(c)
    for r in KNOWN_REGIONS:
        if len(r) <= 2:
            continue
        if r.lower() in t:
            locations.add(r)

    return sorted(companies)[:15], sorted(locations)[:15]


def load_file(fpath, source, fname, max_rows=None):
    """Load a single CSV, detect text column, return cleaned dataframe."""
    df = pd.read_csv(
        fpath, nrows=max_rows, on_bad_lines="skip",
        encoding="utf-8", encoding_errors="ignore",
    )

    str_cols = [str(c) for c in df.columns]
    if all(c.isdigit() for c in str_cols) or \
       (len(df.columns) == 2 and detect_text_column(df) is None):
        df = pd.read_csv(
            fpath, header=None, names=["sentiment", "text"],
            nrows=max_rows, on_bad_lines="skip",
            encoding="utf-8", encoding_errors="ignore",
        )

    has_title = any("title" in str(c).lower() for c in df.columns)
    has_selftext = any("selftext" in str(c).lower() for c in df.columns)

    if has_title and has_selftext:
        title_col = next(c for c in df.columns if "title" in str(c).lower())
        selftext_col = next(c for c in df.columns if "selftext" in str(c).lower())
        df["_text"] = (
            df[title_col].fillna("").astype(str) + " " +
            df[selftext_col].fillna("").astype(str)
        )
        df["_text"] = df["_text"].str.replace(
            r"\[deleted\]|\[removed\]", "", regex=True
        ).str.strip()
    else:
        text_col = detect_text_column(df)
        if text_col is None:
            if "Origin_Port" in df.columns and "Destination_Port" in df.columns:
                def _synth(row):
                    parts = []
                    for c in ["Product_Category", "Origin_Port",
                               "Destination_Port", "Transport_Mode",
                               "Weather_Condition"]:
                        if c in df.columns and pd.notna(row.get(c)):
                            parts.append(str(row[c]))
                    status = "disrupted" if row.get("Disruption_Occurred", 0) == 1 else "on-track"
                    return f"Shipment {status}: {' / '.join(parts)}"
                df["_text"] = df.apply(_synth, axis=1)
            else:
                return pd.DataFrame()
        else:
            df["_text"] = df[text_col].fillna("").astype(str).str.strip()

    df = df[df["_text"].str.len() > 10].copy()
    df["_source"] = source
    df["_file"] = fname
    return df[["_text", "_source", "_file"]]


def main():
    """Run full preprocessing pipeline with BERT NER entity extraction."""
    t0 = time.time()
    print("=" * 70)
    print("  SupplyMind AI — Data Preprocessing Pipeline (NLP Enhanced)")
    print("=" * 70)

    all_rows = []
    file_map = {
        "news": NEWS_FILES,
        "social": SOCIAL_FILES,
        "supply": SUPPLY_FILES,
    }
    for source, file_list in file_map.items():
        for fname in file_list:
            fpath = os.path.join(DATA_DIR, fname)
            if not os.path.exists(fpath):
                print(f"  [SKIP] {fname} not found")
                continue
            max_rows = 50_000 if "wallstreet_bets" in fname else None
            df = load_file(fpath, source, fname, max_rows=max_rows)
            if df.empty:
                print(f"  [SKIP] {fname} — no text column")
                continue

            before = len(df)
            if source == "social":
                mask = (
                    df["_text"].apply(lambda t: relevance_score(t) >= 3) &
                    ~df["_text"].apply(lambda t: noise_score(t) >= 2)
                )
                df = df[mask]
            elif source == "news":
                df = df[df["_text"].apply(lambda t: relevance_score(t) >= 1)]

            after = len(df)
            print(f"  [OK]   {fname}: {before:,} -> {after:,} rows")
            all_rows.append(df)

    if not all_rows:
        print("[ERROR] No data loaded.")
        return

    data = pd.concat(all_rows, ignore_index=True)
    print(f"\n  Total after filtering: {len(data):,}")

    # ── Step 1: Risk scoring ────────────────────────────────────────────────
    print("\n[STEP 1/3] Scoring risk (keyword engine)...")
    texts = data["_text"].tolist()
    results = score_batch(texts)
    data["risk_label"] = [r["label"] for r in results]
    data["risk_score"] = [r["score"] for r in results]
    data["risk_level"] = [r["risk_level"] for r in results]
    data["raw_score"] = [r["raw_score"] for r in results]
    data["signal_count"] = [r["signal_count"] for r in results]

    before_filter = len(data)
    data = data[data["raw_score"] > 0].reset_index(drop=True)
    print(f"  Removed {before_filter - len(data):,} zero-score rows")
    print(f"  Kept {len(data):,} scored rows")

    texts = data["_text"].tolist()

    # ── Step 2: Entity extraction with BERT NER ─────────────────────────────
    print("\n[STEP 2/3] Extracting entities (BERT NER + keywords)...")
    ner_pipeline = try_load_ner()

    companies_list = []
    locations_list = []
    n = len(texts)
    for i, text in enumerate(texts):
        comps, locs = extract_entities_ner(text, ner_pipeline)
        companies_list.append(",".join(comps))
        locations_list.append(",".join(locs))
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {i+1:,}/{n:,} ({(i+1)/n*100:.0f}%) — {elapsed:.0f}s elapsed")

    data["companies"] = companies_list
    data["locations"] = locations_list

    # ── Step 3: Stats ───────────────────────────────────────────────────────
    print("\n[STEP 3/3] Final statistics...")
    print("\n  Risk Distribution:")
    for label in ["critical risk", "high risk", "medium risk", "low risk"]:
        count = int((data["risk_label"] == label).sum())
        pct = count / len(data) * 100
        print(f"    {label:14s}  {count:6,}  ({pct:5.1f}%)")

    unique_comps = set()
    for c in data["companies"]:
        if c:
            unique_comps.update(x.strip() for x in c.split(",") if x.strip())
    unique_locs = set()
    for l in data["locations"]:
        if l:
            unique_locs.update(x.strip() for x in l.split(",") if x.strip())

    print(f"\n  Unique companies detected:  {len(unique_comps):,}")
    print(f"  Unique regions detected:    {len(unique_locs):,}")
    print(f"  Rows with companies:        {(data['companies'].str.len() > 0).sum():,}")
    print(f"  Rows with locations:        {(data['locations'].str.len() > 0).sum():,}")

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = os.path.join(DATA_DIR, "analyzed_data.parquet")
    try:
        data.to_parquet(out_path, index=False)
        print(f"\n  [SAVED] {out_path} ({len(data):,} rows)")
    except Exception as e:
        print(f"\n  [WARN] Parquet failed ({e}), CSV only")

    csv_path = os.path.join(DATA_DIR, "analyzed_data.csv")
    data.to_csv(csv_path, index=False)
    print(f"  [SAVED] {csv_path}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  DONE — {len(data):,} documents processed in {elapsed:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
