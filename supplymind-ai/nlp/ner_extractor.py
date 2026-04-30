"""Named entity recognition using dslim/bert-base-NER on GPU with CPU fallback."""

import os
import sys
import re
import torch
from transformers import pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import NER_MODEL, DEVICE, KNOWN_COMPANIES, KNOWN_REGIONS

_ner_pipeline = None

JUNK_WORDS = {
    "real", "bill", "the", "new", "all", "big", "top", "best", "free", "just",
    "get", "buy", "sell", "long", "short", "call", "put", "yolo", "moon",
    "hold", "gain", "loss", "bear", "bull", "dd", "tldr", "imo", "fyi",
    "and", "for", "not", "but", "are", "was", "has", "had", "his", "her",
    "its", "our", "can", "may", "will", "now", "day", "week", "year",
    "mot", "usa", "gdp", "sec", "fed", "ipo", "ceo", "cfo", "etf",
    "op", "edit", "update", "deleted", "removed", "entertainment",
    "and entertainment", "news", "finance", "market", "stock", "stocks",
    "trading", "investment", "investor", "money", "cash",
    "air", "electronics", "general", "national", "international", "global",
    "north", "south", "east", "west", "central", "united", "states",
    "first", "second", "third", "last", "next", "today", "yesterday",
    "robinhood", "fidelity", "reddit", "wsb", "theta", "gang",
    "options", "calls", "puts", "shares", "futures", "bonds",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "december",
}

JUNK_LOCATIONS = {
    "air", "electronics", "north", "south", "east", "west", "central",
    "general", "united", "states", "national", "international",
    "global", "local", "online", "home", "world", "earth",
}


def get_ner(device=None):
    """Load or return cached NER pipeline."""
    global _ner_pipeline
    if _ner_pipeline is None:
        dev = device or DEVICE
        dev_id = 0 if dev == "cuda" else -1
        print(f"[NLP] Loading NER model on {dev.upper()} ...")
        _ner_pipeline = pipeline(
            "ner",
            model=NER_MODEL,
            aggregation_strategy="simple",
            device=dev_id,
        )
        print("[NLP] NER model ready.")
    return _ner_pipeline


def _is_valid_entity(word, min_len=3, min_score=0.5, score=1.0):
    """Filter out junk NER extractions."""
    if len(word) < min_len:
        return False
    if word.lower() in JUNK_WORDS:
        return False
    if re.match(r"^[^a-zA-Z]*$", word):
        return False
    if score < min_score:
        return False
    return True


def extract_entities(text):
    """Extract companies (ORG) and locations (LOC/GPE) from text, max 10 each."""
    if not text or len(str(text).strip()) < 3:
        return {"companies": [], "locations": []}

    text = str(text)[:512]
    ner = get_ner()
    entities = ner(text)

    companies = set()
    locations = set()

    for ent in entities:
        label = ent.get("entity_group", "")
        word = ent.get("word", "").strip().replace("##", "")
        score = ent.get("score", 0.0)
        if not _is_valid_entity(word, score=score):
            continue
        if label == "ORG":
            companies.add(word)
        elif label in ("LOC", "GPE"):
            if word.lower() not in JUNK_LOCATIONS:
                locations.add(word)

    text_upper = text.upper()
    for company in KNOWN_COMPANIES:
        if company.upper() in text_upper:
            companies.add(company)
    for region in KNOWN_REGIONS:
        if region.upper() in text_upper:
            locations.add(region)

    return {
        "companies": sorted(companies)[:10],
        "locations": sorted(locations)[:10],
    }


def extract_batch(texts):
    """Extract entities from a list of texts."""
    return [extract_entities(t) for t in texts]


if __name__ == "__main__":
    sample = "Apple and TSMC face disruptions in Taiwan due to earthquake near Shenzhen"
    result = extract_entities(sample)
    print(f"Companies: {result['companies']}")
    print(f"Locations: {result['locations']}")
