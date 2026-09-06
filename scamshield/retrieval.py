from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

from scamshield.config import settings

KNOWLEDGE_FILE = Path(__file__).resolve().parent.parent / "knowledge" / "sources.json"


def _tokenize(text: str) -> list[str]:
    return [tok for tok in re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE) if len(tok) > 1]


@lru_cache(maxsize=1)
def load_parent_documents() -> list[dict]:
    docs = json.loads(KNOWLEDGE_FILE.read_text(encoding="utf-8"))
    if not isinstance(docs, list) or not docs:
        raise RuntimeError("knowledge/sources.json must contain a non-empty JSON array.")
    return docs


@lru_cache(maxsize=1)
def load_child_chunks() -> list[dict]:
    chunks: list[dict] = []
    for parent in load_parent_documents():
        for section in parent.get("sections", []):
            chunks.append(
                {
                    "id": f"{parent['id']}::{section['id']}",
                    "parent_id": parent["id"],
                    "title": parent["title"],
                    "organization": parent.get("organization", ""),
                    "country": parent.get("country", "General"),
                    "primary_topic": parent.get("primary_topic", "general"),
                    "tags": parent.get("tags", []),
                    "source": parent["source"],
                    "section_title": section.get("title", section["id"]),
                    "text": section["text"],
                    "parent_summary": parent.get("summary", ""),
                    "reviewed_at": parent.get("reviewed_at", ""),
                }
            )
    if not chunks:
        raise RuntimeError("No RAG sections found in knowledge/sources.json.")
    return chunks


@lru_cache(maxsize=1)
def _parent_by_id() -> dict[str, dict]:
    return {doc["id"]: doc for doc in load_parent_documents()}


def _index_text(chunk: dict) -> str:
    return "\n".join(
        [
            chunk.get("title", ""),
            chunk.get("section_title", ""),
            chunk.get("organization", ""),
            chunk.get("country", ""),
            chunk.get("primary_topic", ""),
            " ".join(chunk.get("tags", [])),
            chunk.get("text", ""),
        ]
    )


@lru_cache(maxsize=1)
def _load_bm25():
    from rank_bm25 import BM25Okapi

    chunks = load_child_chunks()
    corpus = [_tokenize(_index_text(chunk)) for chunk in chunks]
    return BM25Okapi(corpus)


@lru_cache(maxsize=1)
def _load_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model, device=settings.embedding_device)


@lru_cache(maxsize=1)
def _load_chroma():
    import chromadb
    from chromadb.config import Settings

    chunks = load_child_chunks()
    model = _load_embedding_model()
    texts = [_index_text(chunk) for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)

    client = chromadb.Client(Settings(anonymized_telemetry=False))
    collection_name = "scamshield_verified_guidance"
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=texts,
        embeddings=np.asarray(embeddings, dtype=np.float32).tolist(),
        metadatas=[
            {
                "parent_id": chunk["parent_id"],
                "country": chunk["country"],
                "primary_topic": chunk["primary_topic"],
                "organization": chunk["organization"],
            }
            for chunk in chunks
        ],
    )
    return collection, model


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values
    low, high = float(values.min()), float(values.max())
    if math.isclose(low, high):
        return np.ones_like(values) if high > 0 else np.zeros_like(values)
    return (values - low) / (high - low)


def _metadata_boost(chunk: dict, hints: dict | None) -> float:
    if not hints:
        return 0.0
    boost = 0.0
    if hints.get("country") and chunk.get("country") == hints["country"]:
        boost += 0.08
    topics = set(hints.get("topics") or [])
    if chunk.get("primary_topic") in topics:
        boost += 0.10
    if topics.intersection(set(chunk.get("tags", []))):
        boost += 0.05
    return min(boost, 0.18)


def _mmr_order(query_vec: np.ndarray, doc_vecs: np.ndarray, relevance: np.ndarray, k: int) -> list[int]:
    """Maximum Marginal Relevance: balance query relevance with result diversity."""
    if len(doc_vecs) == 0:
        return []
    lam = min(max(settings.rag_mmr_lambda, 0.0), 1.0)
    selected: list[int] = []
    candidates = list(range(len(doc_vecs)))
    similarity_matrix = np.asarray(doc_vecs, dtype=np.float32) @ np.asarray(doc_vecs, dtype=np.float32).T

    while candidates and len(selected) < k:
        best_idx = None
        best_score = -10.0
        for idx in candidates:
            diversity_penalty = max((float(similarity_matrix[idx, j]) for j in selected), default=0.0)
            score = lam * float(relevance[idx]) - (1.0 - lam) * diversity_penalty
            if score > best_score:
                best_score = score
                best_idx = idx
        assert best_idx is not None
        selected.append(best_idx)
        candidates.remove(best_idx)
    return selected


def _semantic_search(query: str, candidate_k: int) -> tuple[dict[str, float], list[str]]:
    collection, model = _load_chroma()
    query_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)[0]
    result = collection.query(
        query_embeddings=[np.asarray(query_vec, dtype=np.float32).tolist()],
        n_results=min(candidate_k, len(load_child_chunks())),
        include=["distances", "embeddings"],
    )
    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]
    embeddings = np.asarray(result.get("embeddings", [[]])[0], dtype=np.float32)
    relevance = np.asarray([max(0.0, 1.0 - float(d)) for d in distances], dtype=np.float32)
    order = _mmr_order(np.asarray(query_vec, dtype=np.float32), embeddings, relevance, len(ids))
    scores = {ids[i]: float(relevance[i]) for i in range(len(ids))}
    mmr_ids = [ids[i] for i in order]
    return scores, mmr_ids


def _keyword_search(query: str, candidate_k: int) -> tuple[dict[str, float], list[str]]:
    chunks = load_child_chunks()
    raw = np.asarray(_load_bm25().get_scores(_tokenize(query)), dtype=np.float32)
    normalized = _normalize_scores(raw)
    indices = np.argsort(normalized)[::-1][:candidate_k]
    scores = {chunks[int(i)]["id"]: float(normalized[i]) for i in indices}
    return scores, [chunks[int(i)]["id"] for i in indices]


def _rrf(rankings: Iterable[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    if not scores:
        return scores
    maximum = max(scores.values())
    return {doc_id: score / maximum for doc_id, score in scores.items()}


def _parent_context(chunk: dict) -> str:
    parent = _parent_by_id()[chunk["parent_id"]]
    sibling_sections = parent.get("sections", [])
    matching = chunk.get("section_title", "")
    extra = []
    for section in sibling_sections:
        if section.get("title") == matching:
            continue
        extra.append(f"{section.get('title', '')}: {section.get('text', '')}")
        if len(extra) >= 1:
            break
    parts = [parent.get("summary", ""), f"{matching}: {chunk['text']}"] + extra
    return "\n".join(part for part in parts if part).strip()


def search_once(query: str, top_k: int | None = None, hints: dict | None = None) -> list[dict]:
    chunks = load_child_chunks()
    by_id = {chunk["id"]: chunk for chunk in chunks}
    top_k = top_k or settings.rag_top_k
    candidate_k = max(top_k * 2, settings.rag_candidate_k)
    backend = settings.rag_backend

    semantic_scores: dict[str, float] = {}
    semantic_rank: list[str] = []
    semantic_ok = False
    if backend in {"advanced", "semantic", "hybrid"}:
        try:
            semantic_scores, semantic_rank = _semantic_search(query, candidate_k)
            semantic_ok = True
        except Exception as exc:
            print(f"[ScamShield RAG] semantic/Chroma unavailable; using keyword fallback: {exc}")

    keyword_scores, keyword_rank = _keyword_search(query, candidate_k)
    rankings = [keyword_rank] + ([semantic_rank] if semantic_rank else [])
    rrf_scores = _rrf(rankings)
    candidate_ids = list(dict.fromkeys(semantic_rank + keyword_rank))

    scored: list[tuple[float, str, dict]] = []
    for doc_id in candidate_ids:
        chunk = by_id[doc_id]
        sem = semantic_scores.get(doc_id, 0.0)
        kw = keyword_scores.get(doc_id, 0.0)
        rrf = rrf_scores.get(doc_id, 0.0)
        boost = _metadata_boost(chunk, hints)
        if semantic_ok:
            score = 0.58 * sem + 0.24 * kw + 0.13 * rrf + boost
        else:
            score = 0.72 * kw + 0.20 * rrf + boost
        scored.append((min(float(score), 1.0), doc_id, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    filtered = [item for item in scored if item[0] >= settings.rag_score_threshold]
    if not filtered:
        filtered = scored[: max(1, min(top_k, len(scored)))]

    results: list[dict] = []
    seen_parents: set[str] = set()
    for score, doc_id, chunk in filtered:
        # Parent-level diversity: avoid returning four sections from one source unless necessary.
        if chunk["parent_id"] in seen_parents and len(seen_parents) >= 2:
            continue
        seen_parents.add(chunk["parent_id"])
        results.append(
            {
                "id": doc_id,
                "parent_id": chunk["parent_id"],
                "title": chunk["title"],
                "organization": chunk["organization"],
                "country": chunk["country"],
                "primary_topic": chunk["primary_topic"],
                "section_title": chunk["section_title"],
                "source": chunk["source"],
                "text": chunk["text"],
                "context": _parent_context(chunk),
                "retrieval_score": round(score, 4),
                "retrieval_backend": "chroma+hf+bm25+mmr+rrf" if semantic_ok else "bm25-fallback",
                "reviewed_at": chunk["reviewed_at"],
            }
        )
        if len(results) >= top_k:
            break
    return results


def retrieve_guidance(
    query: str,
    *,
    top_k: int | None = None,
    hints: dict | None = None,
    query_variants: list[str] | None = None,
) -> tuple[list[dict], float, list[str]]:
    """Advanced RAG with hybrid retrieval, MMR, metadata boosts, fusion, and optional multi-query."""
    variants = [query] + [q for q in (query_variants or []) if q.strip() and q.strip() != query.strip()]
    per_query = [search_once(q, top_k=max(settings.rag_top_k, 5), hints=hints) for q in variants]
    strategy = ["Hugging Face multilingual embeddings", "ChromaDB semantic search", "BM25 keyword search", "MMR diversity", "RRF fusion", "metadata-aware boosting", "parent context"]
    if len(variants) > 1:
        strategy.append("conditional multi-query expansion")

    if len(per_query) == 1:
        results = per_query[0][:(top_k or settings.rag_top_k)]
    else:
        rankings = [[item["id"] for item in result] for result in per_query]
        fusion = _rrf(rankings)
        best_by_id: dict[str, dict] = {}
        for result in per_query:
            for item in result:
                previous = best_by_id.get(item["id"])
                if previous is None or item["retrieval_score"] > previous["retrieval_score"]:
                    best_by_id[item["id"]] = dict(item)
        for doc_id, item in best_by_id.items():
            item["retrieval_score"] = round(0.70 * item["retrieval_score"] + 0.30 * fusion.get(doc_id, 0.0), 4)
        results = sorted(best_by_id.values(), key=lambda x: x["retrieval_score"], reverse=True)
        # Prefer parent diversity in final fused set.
        final: list[dict] = []
        parents: set[str] = set()
        for item in results:
            if item["parent_id"] in parents and len(parents) >= 2:
                continue
            final.append(item)
            parents.add(item["parent_id"])
            if len(final) >= (top_k or settings.rag_top_k):
                break
        results = final

    confidence = float(results[0]["retrieval_score"]) if results else 0.0
    return results, round(confidence, 4), strategy


def rag_health() -> dict:
    return {
        "parent_documents": len(load_parent_documents()),
        "indexed_sections": len(load_child_chunks()),
        "backend": settings.rag_backend,
        "embedding_model": settings.embedding_model,
        "top_k": settings.rag_top_k,
        "score_threshold": settings.rag_score_threshold,
        "expand_threshold": settings.rag_expand_threshold,
    }
