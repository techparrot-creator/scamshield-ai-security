"""ScamShield multilingual benchmark evaluator (LIVE Gemini calls).

Runs the FULL ScamShield LangGraph + advanced RAG pipeline over the benchmark in
``data/evaluation/`` and reports classification metrics with breakdowns. This is
separate from the RAG retrieval eval and from the RAG knowledge base.

Label mapping (documented, no fabricated metrics):
- assessment risk_level high/critical -> predicted "scam"
- low                                 -> predicted "legitimate"
- medium                              -> predicted "uncertain" (calibrated abstention)
- no assessment                       -> predicted "unavailable"
Ambiguous-expected cases and uncertain/unavailable predictions are EXCLUDED from
binary metrics and reported separately. Screenshot/audio cases without a real
fixture file are reported as "fixture_unavailable" and never counted as a PASS.

Exports machine-readable results (JSON + CSV) under data/evaluation/results/.

Usage:
    python scripts/evaluate_benchmark.py                    # development split
    python scripts/evaluate_benchmark.py --split holdout    # locked holdout
    python scripts/evaluate_benchmark.py --limit 10
    python scripts/evaluate_benchmark.py --no-export
    # Balanced, reproducible 30-case executable holdout sample:
    python scripts/evaluate_benchmark.py --split holdout --stratified --executable-only --limit 30 --seed 20260904
"""
from __future__ import annotations

import argparse
import csv
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
    DEFAULT_SAMPLE_SEED,
    is_executable,
    run_all,
    run_case_full,
    select_cases,
    stratified_sample,
)

DEFAULT_DATASET = ROOT / "data" / "evaluation" / "scamshield_benchmark.jsonl"
RESULTS_DIR = ROOT / "data" / "evaluation" / "results"

CSV_COLUMNS = [
    "case_id", "language", "category", "modality", "difficulty",
    "expected_label", "predicted", "risk_level", "status",
]


def _fmt(value) -> str:
    return f"{value:.1%}" if isinstance(value, float) else "n/a"


def print_summary(summary: dict) -> None:
    counts, metrics = summary["counts"], summary["metrics"]
    print("\n=== Counts ===")
    print(f"Total                : {summary['total']}")
    print(f"Executed             : {summary['executed']} (entered the model; excludes missing fixtures)")
    print(f"Execution coverage   : {_fmt(summary['execution_coverage'])} (executed / total)")
    print(f"Model responded      : {summary['model_responded']} (returned a risk band)")
    print(f"Definitive (scored)  : {summary['definitive']}")
    print(f"Ambiguous expected   : {summary['ambiguous_expected']} (excluded from binary metrics)")
    print(f"Predicted uncertain  : {summary['predicted_uncertain']} (real model abstention, excluded from binary)")
    print(f"Predicted unavailable: {summary['predicted_unavailable']} (model failure/timeout, excluded)")
    print(f"Fixture unavailable  : {summary['fixture_unavailable']} (media with no real fixture; NOT executed, NOT a model failure)")

    print("\n=== Confusion matrix (rows = expected, cols = predicted; definitive only) ===")
    print(f"{'':<14}{'pred scam':>12}{'pred legit':>12}")
    print(f"{'expected scam':<14}{counts['tp']:>12}{counts['fn']:>12}")
    print(f"{'expected legit':<14}{counts['fp']:>12}{counts['tn']:>12}")

    print("\n=== Metrics ===")
    print(f"TP / TN / FP / FN    : {counts['tp']} / {counts['tn']} / {counts['fp']} / {counts['fn']}")
    print(f"Accuracy             : {_fmt(metrics['accuracy'])}")
    print(f"Scam recall          : {_fmt(metrics['scam_recall'])}")
    print(f"Legitimate specificity: {_fmt(metrics['legitimate_specificity'])}")
    print(f"False-positive rate  : {_fmt(metrics['false_positive_rate'])}")
    print(f"Precision            : {_fmt(metrics['precision'])}")
    print(f"F1                   : {_fmt(metrics['f1'])}")
    print(f"Uncertainty/abstention rate: {_fmt(summary['abstention_rate'])} (uncertain / model-responded only)")

    for field in ("language", "category", "modality", "difficulty"):
        print(f"\n=== Breakdown by {field} (scored cases) ===")
        for key, row in summary["breakdowns"][field].items():
            print(
                f"{key:<26} total={row['total']:<4} scored={row['scored']:<4} exec={row['executed']:<4} "
                f"acc={_fmt(row['accuracy'])} recall={_fmt(row['scam_recall'])} "
                f"spec={_fmt(row['legitimate_specificity'])} fpr={_fmt(row['false_positive_rate'])} "
                f"uncert={row['uncertain']} model_unavail={row['unavailable']} "
                f"fixture_unavail={row['fixture_unavailable']}"
            )


def export_results(summary: dict, runs: list[dict], split: str, tag: str) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = RESULTS_DIR / f"benchmark_{tag}_{split}_{stamp}.json"
    csv_path = RESULTS_DIR / f"benchmark_{tag}_{split}_{stamp}.csv"
    json_path.write_text(
        json.dumps({"split": split, "tag": tag, "generated_utc": stamp, "summary": summary, "runs": runs},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for run in runs:
            writer.writerow(run)
    return json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="ScamShield benchmark evaluator (live Gemini).")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Benchmark JSONL path")
    parser.add_argument("--split", default="development", choices=["development", "holdout", "all"],
                        help="Which split to evaluate (holdout is locked; do not tune on it)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Sample size N (0 = all); with --stratified this is the balanced sample size")
    parser.add_argument("--stratified", action="store_true",
                        help="Deterministic sample balanced across language, expected label, and modality")
    parser.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED,
                        help="Seed for reproducible --stratified sampling")
    parser.add_argument("--executable-only", action="store_true",
                        help="Exclude screenshot/audio cases whose real fixture file is unavailable")
    parser.add_argument("--delay", type=float, default=0.25, help="Seconds between live Gemini calls")
    parser.add_argument("--no-export", action="store_true", help="Do not write JSON/CSV results")
    parser.add_argument("--quiet", action="store_true", help="Do not print per-case rows")
    args = parser.parse_args()

    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print("BLOCKED: set GOOGLE_API_KEY in .env before running the benchmark evaluator (live Gemini calls).")
        return 2
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"BLOCKED: benchmark not found: {dataset_path}. Run scripts/build_benchmark.py first.")
        return 2

    cases = select_cases(load_jsonl(dataset_path), args.split)

    # Executable-only filtering happens BEFORE sampling so it applies to both the
    # stratified and the plain first-N paths. Missing-fixture media are dropped up
    # front instead of being run and reported as fixture_unavailable.
    if args.executable_only:
        before = len(cases)
        cases = [c for c in cases if is_executable(c, ROOT)]
        dropped = before - len(cases)
        if dropped:
            print(f"Executable-only: excluded {dropped} screenshot/audio case(s) with no real fixture file.")

    if args.stratified:
        cases = stratified_sample(cases, args.limit, seed=args.seed)
    elif args.limit > 0:
        cases = cases[: args.limit]

    if not cases:
        print("BLOCKED: no cases selected for this split.")
        return 2

    media_pending = sum(1 for c in cases if c["modality"] in {"screenshot", "audio"})
    print(f"Benchmark evaluation: split={args.split}, {len(cases)} cases (live Gemini calls).")
    if args.stratified or args.executable_only:
        flags = []
        if args.stratified:
            flags.append(f"stratified sample (seed={args.seed}, size={args.limit or 'all'})")
        if args.executable_only:
            flags.append("executable-only")
        print("Sampling: " + ", ".join(flags) + ".")
    print(f"Mapping: high/critical->scam, low->legitimate, medium->uncertain (excluded), missing->unavailable (excluded).")
    if media_pending:
        print(f"Note: {media_pending} screenshot/audio case(s) require a real fixture file; without one they are")
        print("      reported as fixture_unavailable and are NEVER counted as a multimodal PASS.")
    print()

    runs = run_all(cases, ROOT, run_case_full, delay=args.delay, verbose=not args.quiet)
    summary = summarize_runs(runs)
    print_summary(summary)

    if not args.no_export:
        json_path, csv_path = export_results(summary, runs, args.split, "full")
        print(f"\nExported:\n  {json_path.relative_to(ROOT)}\n  {csv_path.relative_to(ROOT)}")

    if summary["definitive"] == 0:
        print("\nWARNING: no case produced a definitive mapping; binary metrics are not meaningful.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
