"""Baseline vs full-pipeline ablation on the LOCKED holdout split (LIVE Gemini).

Compares, on the SAME benchmark cases:
  A. Gemini-only baseline: raw evidence -> single structured assessment (no RAG,
     no URL heuristics, no query planning, no LangGraph).
  B. Full ScamShield: the complete LangGraph + advanced hybrid RAG pipeline.

Both are scored with the identical documented mapping and metrics. Results are
exported to JSON. Nothing is fabricated: if API quota/errors prevent completion,
the run is reported as INCOMPLETE with the exact executed counts, and the process
exits non-zero so the shortfall is visible.

Usage:
    python scripts/evaluate_baseline_ablation.py                 # holdout split
    python scripts/evaluate_baseline_ablation.py --split development
    python scripts/evaluate_baseline_ablation.py --limit 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from scamshield.benchmark import load_jsonl, summarize_runs  # noqa: E402
from scamshield.benchmark_runner import (  # noqa: E402
    run_all,
    run_case_baseline,
    run_case_full,
    select_cases,
)

DEFAULT_DATASET = ROOT / "data" / "evaluation" / "scamshield_benchmark.jsonl"
RESULTS_DIR = ROOT / "data" / "evaluation" / "results"


def _fmt(value) -> str:
    return f"{value:.1%}" if isinstance(value, float) else "n/a"


def print_comparison(base_summary: dict, full_summary: dict) -> None:
    print("\n=== Ablation: A) Gemini-only baseline   B) Full ScamShield ===")
    print(f"{'Metric':<28}{'A baseline':>14}{'B full':>14}")
    print("-" * 56)
    for label, key in (
        ("Total", "total"),
        ("Definitive (scored)", "definitive"),
        ("Ambiguous expected", "ambiguous_expected"),
        ("Uncertain", "predicted_uncertain"),
        ("Unavailable", "predicted_unavailable"),
        ("Fixture unavailable", "fixture_unavailable"),
    ):
        print(f"{label:<28}{base_summary[key]:>14}{full_summary[key]:>14}")
    bc, fc = base_summary["counts"], full_summary["counts"]
    print(f"{'TP/TN/FP/FN':<28}"
          f"{str(bc['tp'])+'/'+str(bc['tn'])+'/'+str(bc['fp'])+'/'+str(bc['fn']):>14}"
          f"{str(fc['tp'])+'/'+str(fc['tn'])+'/'+str(fc['fp'])+'/'+str(fc['fn']):>14}")
    for name in ("accuracy", "scam_recall", "legitimate_specificity", "false_positive_rate", "precision", "f1"):
        print(f"{name:<28}{_fmt(base_summary['metrics'][name]):>14}{_fmt(full_summary['metrics'][name]):>14}")
    print(f"{'abstention_rate':<28}{_fmt(base_summary['abstention_rate']):>14}{_fmt(full_summary['abstention_rate']):>14}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline vs full ScamShield ablation (live Gemini).")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--split", default="holdout", choices=["development", "holdout", "all"],
                        help="Locked holdout by default; do NOT tune thresholds on it.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print("BLOCKED: set GOOGLE_API_KEY in .env before running the ablation (live Gemini calls).")
        return 2
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"BLOCKED: benchmark not found: {dataset_path}. Run scripts/build_benchmark.py first.")
        return 2

    cases = select_cases(load_jsonl(dataset_path), args.split)
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        print("BLOCKED: no cases selected for this split.")
        return 2

    print(f"Ablation on split='{args.split}': {len(cases)} cases. Running BOTH paths on the identical set.")
    media_pending = sum(1 for c in cases if c["modality"] in {"screenshot", "audio"})
    if media_pending:
        print(f"Note: {media_pending} screenshot/audio case(s) need a real fixture; without one they are")
        print("      fixture_unavailable for BOTH paths (multimodal is never claimed as executed).")

    print("\n--- A) Gemini-only baseline ---")
    base_runs = run_all(cases, ROOT, run_case_baseline, delay=args.delay, verbose=not args.quiet)
    print("\n--- B) Full ScamShield ---")
    full_runs = run_all(cases, ROOT, run_case_full, delay=args.delay, verbose=not args.quiet)

    base_summary = summarize_runs(base_runs)
    full_summary = summarize_runs(full_runs)
    print_comparison(base_summary, full_summary)

    # Incomplete-execution detection: an unavailable/errored call means quota or outage.
    base_incomplete = sum(1 for r in base_runs if r["status"].startswith(("baseline_error", "error", "media_error")))
    full_incomplete = sum(1 for r in full_runs if r["status"].startswith(("error", "media_error")))
    incomplete = base_incomplete + full_incomplete

    if not args.no_export:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RESULTS_DIR / f"ablation_{args.split}_{stamp}.json"
        out_path.write_text(
            json.dumps(
                {
                    "split": args.split,
                    "generated_utc": stamp,
                    "case_count": len(cases),
                    "executed": {
                        "baseline": sum(1 for r in base_runs if not r["status"].endswith("unavailable")),
                        "full": sum(1 for r in full_runs if not r["status"].endswith("unavailable")),
                    },
                    "incomplete_calls": {"baseline": base_incomplete, "full": full_incomplete},
                    "baseline_summary": base_summary,
                    "full_summary": full_summary,
                    "baseline_runs": base_runs,
                    "full_runs": full_runs,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nExported: {out_path.relative_to(ROOT)}")

    if incomplete:
        print(f"\nINCOMPLETE EXECUTION: {incomplete} live call(s) errored (likely API quota/outage).")
        print("Metrics above reflect only the cases that actually executed; they are NOT a full ablation.")
        return 1
    if base_summary["definitive"] == 0 or full_summary["definitive"] == 0:
        print("\nWARNING: one path produced no definitive mappings; comparison is not meaningful.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
