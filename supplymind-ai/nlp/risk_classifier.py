"""Zero-shot risk classification using facebook/bart-large-mnli on GPU with CPU fallback."""

import os
import sys
import time
import torch
from transformers import pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import ZERO_SHOT_MODEL, RISK_LABELS, DEVICE, BATCH_SIZE

RISK_LEVEL_MAP = {
    "low risk": 1,
    "medium risk": 2,
    "high risk": 3,
    "critical risk": 4,
}

_classifier = None


def get_classifier(device=None):
    """Load or return cached zero-shot classification pipeline."""
    global _classifier
    if _classifier is None:
        dev = device or DEVICE
        dev_id = 0 if dev == "cuda" else -1
        print(f"[NLP] Loading zero-shot classifier on {dev.upper()} ...")
        _classifier = pipeline(
            "zero-shot-classification",
            model=ZERO_SHOT_MODEL,
            device=dev_id,
        )
        print("[NLP] Zero-shot classifier ready.")
    return _classifier


def classify_risk(text, device=None):
    """Classify a single text into a risk level and return label, score, and level."""
    if not text or len(str(text).strip()) < 3:
        return {"label": "low risk", "score": 0.0, "risk_level": 1}

    clf = get_classifier(device)
    result = clf(str(text)[:512], candidate_labels=RISK_LABELS)
    top_label = result["labels"][0]
    top_score = round(result["scores"][0], 4)
    return {
        "label": top_label,
        "score": top_score,
        "risk_level": RISK_LEVEL_MAP.get(top_label, 1),
    }


def classify_batch(texts, device=None):
    """Classify a list of texts and return a list of result dicts."""
    clf = get_classifier(device)
    results = []
    for text in texts:
        if not text or len(str(text).strip()) < 3:
            results.append({"label": "low risk", "score": 0.0, "risk_level": 1})
            continue
        out = clf(str(text)[:512], candidate_labels=RISK_LABELS)
        results.append({
            "label": out["labels"][0],
            "score": round(out["scores"][0], 4),
            "risk_level": RISK_LEVEL_MAP.get(out["labels"][0], 1),
        })
    return results


def benchmark_gpu_vs_cpu(n_samples=50):
    """Time classification on CPU vs GPU and print the speedup multiplier."""
    sample_texts = [
        "Global semiconductor shortage disrupts auto manufacturing across Asia",
        "Apple reports strong quarterly earnings beating analyst expectations",
        "Severe flooding in Taiwan threatens TSMC chip production facilities",
        "Trade tensions between US and China escalate with new tariffs",
        "Supply chain disruption causes massive delays in European ports",
    ] * (n_samples // 5 + 1)
    sample_texts = sample_texts[:n_samples]

    print(f"\n{'='*60}")
    print(f"  BENCHMARK: Zero-shot Risk Classification ({n_samples} samples)")
    print(f"{'='*60}")

    global _classifier
    _classifier = None
    cpu_pipe = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL, device=-1)
    t0 = time.time()
    for t in sample_texts:
        cpu_pipe(t[:512], candidate_labels=RISK_LABELS)
    cpu_time = time.time() - t0
    print(f"  CPU time: {cpu_time:.2f}s ({n_samples/cpu_time:.1f} samples/sec)")

    gpu_time = None
    speedup = 1.0
    if torch.cuda.is_available():
        gpu_pipe = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL, device=0)
        # warmup
        gpu_pipe(sample_texts[0][:512], candidate_labels=RISK_LABELS)
        t0 = time.time()
        for t in sample_texts:
            gpu_pipe(t[:512], candidate_labels=RISK_LABELS)
        gpu_time = time.time() - t0
        speedup = cpu_time / gpu_time
        print(f"  GPU time: {gpu_time:.2f}s ({n_samples/gpu_time:.1f} samples/sec)")
        print(f"  GPU Speedup: {speedup:.1f}x")
    else:
        print("  GPU: not available — CUDA not detected")

    print(f"{'='*60}\n")
    _classifier = None
    return {"cpu_time": cpu_time, "gpu_time": gpu_time, "speedup": speedup}


if __name__ == "__main__":
    print(classify_risk("TSMC factory shutdown due to earthquake in Taiwan"))
    benchmark_gpu_vs_cpu()
