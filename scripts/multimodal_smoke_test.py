"""Multimodal proof smoke test.

Runs real Gemini multimodal extraction on supplied image/audio evidence and
feeds the extracted text into the SAME LangGraph + advanced RAG pipeline used
by the product UI. Only safe technical results are printed (counts, labels,
source IDs); raw media contents are never printed.

Usage:
    python scripts/multimodal_smoke_test.py --image path/to/sample.png
    python scripts/multimodal_smoke_test.py --audio path/to/sample.wav
    python scripts/multimodal_smoke_test.py --image shot.png --audio note.wav

Requires GOOGLE_API_KEY in .env (live Gemini calls are made).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def print_run(summary: dict) -> bool:
    """Print one safe technical result block and return PASS/FAIL."""
    extraction_ok = bool(summary.get("extraction_succeeded"))
    retrieval_ok = bool(summary.get("retrieval_executed"))
    passed = extraction_ok and retrieval_ok

    print(f"\n=== {summary.get('file', '?')} ===")
    print(f"Modality detected     : {summary.get('modality') or 'unsupported'}")
    if extraction_ok:
        print(f"Extraction            : succeeded ({summary.get('extracted_chars', 0)} chars extracted)")
    else:
        print(f"Extraction            : failed gracefully ({summary.get('error', 'unknown error')})")
    print(f"Retrieval executed    : {'yes' if retrieval_ok else 'no'}")
    confidence = summary.get("retrieval_confidence")
    print(f"Retrieval confidence  : {confidence:.3f}" if isinstance(confidence, (int, float)) else "Retrieval confidence  : n/a")
    print(f"Multi-Query activated : {'yes' if summary.get('multi_query_activated') else 'no'}")
    print(f"Retrieved source IDs  : {', '.join(summary.get('source_ids', [])) or 'none'}")
    print(f"Final assessment      : {'yes' if summary.get('assessment_available') else 'no'} (ai_status={summary.get('ai_status', 'n/a')})")
    print(f"Result                : {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="ScamShield multimodal smoke test (live Gemini).")
    parser.add_argument("--image", action="append", default=[], help="Path to an image evidence file (.png/.jpg/.jpeg/.webp)")
    parser.add_argument("--audio", action="append", default=[], help="Path to an audio evidence file (.wav/.mp3/.m4a/.ogg/.flac)")
    args = parser.parse_args()

    files = list(args.image) + list(args.audio)
    if not files:
        print("BLOCKED: supply at least one file, e.g. --image sample.png or --audio sample.wav")
        return 2
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print("BLOCKED: set GOOGLE_API_KEY in .env before running the multimodal smoke test.")
        return 2

    from scamshield.media import run_media_pipeline

    print("ScamShield multimodal smoke test (Gemini extraction -> same LangGraph/RAG pipeline)")
    results: list[bool] = []
    for raw_path in files:
        path = Path(raw_path)
        if not path.exists():
            print(f"\n=== {path.name} ===")
            print("Extraction            : failed gracefully (file not found)")
            print("Result                : FAIL")
            results.append(False)
            continue
        try:
            summary = run_media_pipeline(path)
        except Exception as exc:  # defensive: the pipeline itself degrades gracefully
            print(f"\n=== {path.name} ===")
            print(f"Extraction            : failed gracefully ({type(exc).__name__})")
            print("Result                : FAIL")
            results.append(False)
            continue
        results.append(print_run(summary))

    if all(results):
        print(f"\nPASS: {len(results)} multimodal evidence path(s) executed end-to-end.")
        return 0
    print(f"\nFAIL: {results.count(False)}/{len(results)} path(s) did not complete.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
