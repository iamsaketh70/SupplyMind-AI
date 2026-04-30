"""Sentence embeddings using all-MiniLM-L6-v2 on CUDA with CPU fallback and benchmark."""

import os
import sys
import time
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import EMBEDDING_MODEL, DEVICE, BATCH_SIZE

_model = None


def get_model(device=None):
    """Load or return cached sentence-transformers model."""
    global _model
    if _model is None:
        dev = device or DEVICE
        print(f"[NLP] Loading embedding model '{EMBEDDING_MODEL}' on {dev.upper()} ...")
        _model = SentenceTransformer(EMBEDDING_MODEL, device=dev)
        print("[NLP] Embedding model ready.")
    return _model


def encode(texts, device=None, batch_size=None):
    """Encode a list of texts into embeddings."""
    model = get_model(device)
    bs = batch_size or BATCH_SIZE
    embeddings = model.encode(
        texts,
        batch_size=bs,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings


def benchmark(n_samples=1000):
    """Encode texts on CPU then GPU and print the speedup."""
    sample_texts = [
        "Supply chain disruption in semiconductor industry",
        "Global shipping delays affect consumer electronics",
        "Trade war impact on technology sector exports",
        "Natural disaster threatens manufacturing plants",
        "New trade agreement boosts cross-border commerce",
    ] * (n_samples // 5 + 1)
    sample_texts = sample_texts[:n_samples]

    print(f"\n{'='*60}")
    print(f"  BENCHMARK: Sentence Embeddings ({n_samples} samples)")
    print(f"{'='*60}")

    global _model
    _model = None

    cpu_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    t0 = time.time()
    cpu_model.encode(sample_texts, batch_size=64, show_progress_bar=False)
    cpu_time = time.time() - t0
    print(f"  CPU time: {cpu_time:.2f}s ({n_samples/cpu_time:.0f} texts/sec)")

    gpu_time = None
    speedup = 1.0
    if torch.cuda.is_available():
        gpu_model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
        gpu_model.encode(sample_texts[:10], batch_size=10, show_progress_bar=False)
        t0 = time.time()
        gpu_model.encode(sample_texts, batch_size=256, show_progress_bar=False)
        gpu_time = time.time() - t0
        speedup = cpu_time / gpu_time
        print(f"  GPU time: {gpu_time:.2f}s ({n_samples/gpu_time:.0f} texts/sec)")
        print(f"  GPU Speedup: {speedup:.1f}x")
    else:
        print("  GPU: not available — CUDA not detected")

    print(f"{'='*60}\n")
    _model = None
    return {"cpu_time": cpu_time, "gpu_time": gpu_time, "speedup": speedup}


if __name__ == "__main__":
    emb = encode(["Test sentence for embeddings"])
    print(f"Embedding shape: {emb.shape}")
    benchmark()
