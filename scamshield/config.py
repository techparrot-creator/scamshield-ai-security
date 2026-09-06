from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    gemini_media_model: str = os.getenv("GEMINI_MEDIA_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"))
    embedding_model: str = os.getenv(
        "HF_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    embedding_device: str = os.getenv("HF_EMBEDDING_DEVICE", "cpu")
    rag_backend: str = os.getenv("RAG_BACKEND", "advanced").lower()
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    rag_candidate_k: int = int(os.getenv("RAG_CANDIDATE_K", "12"))
    rag_score_threshold: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.28"))
    rag_expand_threshold: float = float(os.getenv("RAG_EXPAND_THRESHOLD", "0.43"))
    rag_mmr_lambda: float = float(os.getenv("RAG_MMR_LAMBDA", "0.72"))
    gemini_max_attempts: int = int(os.getenv("GEMINI_MAX_ATTEMPTS", "3"))
    gemini_retry_base_delay: float = float(os.getenv("GEMINI_RETRY_BASE_DELAY", "1.0"))


settings = Settings()
