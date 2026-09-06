from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

REQUIRED_IMPORTS = {
    "langgraph": "langgraph",
    "langchain_core": "langchain-core",
    "langchain_google_genai": "langchain-google-genai",
    "google.genai": "google-genai",
    "gradio": "gradio",
    "sentence_transformers": "sentence-transformers",
    "chromadb": "chromadb",
    "rank_bm25": "rank-bm25",
    "numpy": "numpy",
    "pydantic": "pydantic",
}


def _ok(label: str, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    print(f"[PASS] {label}{suffix}")


def _warn(label: str, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    print(f"[WARN] {label}{suffix}")


def _fail(label: str, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    print(f"[FAIL] {label}{suffix}")


def main() -> int:
    failures = 0
    print("ScamShield environment doctor")
    print(f"Python: {sys.version.split()[0]} | OS: {platform.platform()}")
    print(f"Project root: {ROOT}")

    if (3, 11) <= sys.version_info[:2] < (3, 14):
        _ok("Supported Python version")
    else:
        _fail("Supported Python version", "Use Python 3.11, 3.12, or 3.13")
        failures += 1

    required_files = [
        "app.py",
        "langgraph.json",
        "requirements.txt",
        "knowledge/sources.json",
        "data/eval_cases.jsonl",
        "scamshield/graph.py",
    ]
    for relative in required_files:
        path = ROOT / relative
        if path.exists():
            _ok(f"File {relative}")
        else:
            _fail(f"File {relative}", "missing")
            failures += 1

    for module_name, package_name in REQUIRED_IMPORTS.items():
        try:
            importlib.import_module(module_name)
            _ok(f"Dependency {package_name}")
        except Exception as exc:
            _fail(f"Dependency {package_name}", f"{type(exc).__name__}: {exc}")
            failures += 1

    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        _ok("Gemini API key configured", "value hidden")
    else:
        _warn("Gemini API key not configured", "add GOOGLE_API_KEY to .env for end-to-end tests")

    try:
        from scamshield.retrieval import rag_health

        health = rag_health()
        _ok(
            "RAG knowledge metadata",
            f"{health['parent_documents']} sources / {health['indexed_sections']} sections",
        )
    except Exception as exc:
        _fail("RAG knowledge metadata", f"{type(exc).__name__}: {exc}")
        failures += 1

    print()
    if failures:
        print(f"Doctor result: FAIL ({failures} blocking issue(s))")
        return 1
    print("Doctor result: PASS (external API/network calls not tested)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
