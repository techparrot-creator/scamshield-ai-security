# Using Alibaba Qoder with this repo

Qoder is optional developer tooling, not a runtime dependency of ScamShield. The judges assess what is actually built, so keep the application portable and reproducible.

## Suggested first prompt in Qoder

Read this repository completely before editing. ScamShield AI is an AI-hackathon project using LangGraph, Gemini, Gradio, Hugging Face multilingual embeddings, ChromaDB, BM25, and advanced RAG. Preserve the public API of `app.py` and `scamshield/graph.py`. Never hard-code secrets. Before changing code, explain the files you will touch. After changes, run `pytest -q` and preserve fallback behavior when semantic embeddings cannot load.

## Good Qoder tasks

- add a new official RAG source and its retrieval evaluation case;
- improve Roman Urdu topic classification;
- add a licensed URL-reputation provider behind an environment flag;
- add LangSmith evaluation scripts;
- improve Gradio accessibility and low-literacy mode;
- add unit tests before refactoring retrieval.
