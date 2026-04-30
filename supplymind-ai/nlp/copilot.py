"""Keyword + embedding search for Q&A answering — no external API keys needed."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import KNOWN_COMPANIES


def keyword_search(query, texts, top_k=5):
    """Find the top-k texts most relevant to a query using keyword overlap scoring."""
    query_tokens = set(query.lower().split())
    scored = []
    for i, text in enumerate(texts):
        text_lower = text.lower()
        score = sum(1 for token in query_tokens if token in text_lower)
        if score > 0:
            scored.append((score, i, text))
    scored.sort(key=lambda x: -x[0])
    return scored[:top_k]


def embedding_search(query, texts, embeddings, top_k=5):
    """Find the top-k texts closest to the query using cosine similarity on embeddings."""
    from nlp.embedder import encode

    query_emb = encode([query])
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = np.dot(normed, query_emb.T).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(float(similarities[i]), i, texts[i]) for i in top_indices]


def search(query, texts, risk_labels=None, entities=None, embeddings=None, top_k=5):
    """Combined search: keyword + optional embedding search, with summarization."""
    if embeddings is not None and len(embeddings) == len(texts):
        results = embedding_search(query, texts, embeddings, top_k)
    else:
        results = keyword_search(query, texts, top_k)

    if not results:
        return {
            "found": 0,
            "summary": "No relevant documents found for your query.",
            "results": [],
        }

    high_critical = 0
    mentioned_companies = set()

    for _, idx, text in results:
        if risk_labels and idx < len(risk_labels):
            label = risk_labels[idx]
            if label in ("high risk", "critical risk"):
                high_critical += 1
        if entities and idx < len(entities):
            ent = entities[idx]
            if isinstance(ent, dict):
                mentioned_companies.update(ent.get("companies", []))
        for company in KNOWN_COMPANIES:
            if company.lower() in text.lower():
                mentioned_companies.add(company)

    summary_parts = [f"Found {len(results)} relevant documents."]
    if high_critical > 0:
        summary_parts.append(f"{high_critical} are HIGH or CRITICAL risk.")
    if mentioned_companies:
        summary_parts.append(f"Companies mentioned: {', '.join(sorted(mentioned_companies)[:8])}.")

    formatted_results = []
    for score, idx, text in results:
        entry = {"score": score, "index": idx, "text": text[:300]}
        if risk_labels and idx < len(risk_labels):
            entry["risk_label"] = risk_labels[idx]
        formatted_results.append(entry)

    return {
        "found": len(results),
        "high_critical_count": high_critical,
        "companies": sorted(mentioned_companies),
        "summary": " ".join(summary_parts),
        "results": formatted_results,
    }


if __name__ == "__main__":
    sample_texts = [
        "TSMC production halted in Taiwan due to earthquake",
        "Apple reports record revenue in Q4 earnings call",
        "Global semiconductor shortage worsens supply chain",
        "Samsung factory in Shenzhen resumes operations",
        "Trade tensions between US and China impact exports",
    ]
    result = search("Why is Taiwan risk high?", sample_texts)
    print(f"Summary: {result['summary']}")
    for r in result["results"]:
        print(f"  [{r['score']}] {r['text'][:80]}...")
