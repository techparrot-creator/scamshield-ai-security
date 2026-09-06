"""Dataset-quality report for the ScamShield benchmark (offline, no API calls).

Detects exact duplicates, near duplicates, label contradictions, language and
category imbalance, and repeated templates. Prints a human-readable report and
exports JSON to data/evaluation/results/.

Usage:
    python scripts/benchmark_quality_report.py
    python scripts/benchmark_quality_report.py --dataset data/evaluation/scamshield_benchmark.jsonl
    python scripts/benchmark_quality_report.py --threshold 0.92
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scamshield.benchmark import (  # noqa: E402
    LANGUAGES,
    distributions,
    load_jsonl,
    quality_report,
    validate_case,
)

DEFAULT_DATASET = ROOT / "data" / "evaluation" / "scamshield_benchmark.jsonl"
RESULTS_DIR = ROOT / "data" / "evaluation" / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description="ScamShield benchmark dataset-quality report.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--threshold", type=float, default=0.94, help="Near-duplicate similarity threshold")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"BLOCKED: benchmark not found: {dataset_path}. Run scripts/build_benchmark.py first.")
        return 2

    cases = load_jsonl(dataset_path)
    dist = distributions(cases)
    report = quality_report(cases, near_dup_threshold=args.threshold)

    schema_problems = [(c["case_id"], p) for c in cases for p in validate_case(c)]

    print("=== ScamShield benchmark dataset-quality report ===")
    print(f"Dataset              : {dataset_path.relative_to(ROOT)}")
    print(f"Total cases          : {report['total']}")
    print(f"Distinct categories  : {report['distinct_category_count']}")

    print("\n--- Schema ---")
    print(f"Schema problems      : {len(schema_problems)}")
    for case_id, problem in schema_problems[:10]:
        print(f"  - {case_id}: {problem}")

    print("\n--- Duplicates / templates ---")
    print(f"Exact duplicate groups: {report['exact_duplicate_groups']}")
    print(f"Near-duplicate pairs (>={args.threshold}): {report['near_duplicate_pair_count']}")
    for id_a, id_b, ratio in report["near_duplicate_pairs"][:10]:
        print(f"  - {id_a} ~ {id_b} = {ratio}")
    print(f"Repeated templates (>{3}x within a language): {report['repeated_template_count']}")
    for key, count in list(report["repeated_templates"].items())[:10]:
        print(f"  - {count}x  {key[:80]}")

    print("\n--- Label integrity ---")
    print(f"Label contradictions : {report['label_contradiction_count']}")
    for text, l1, l2 in report["label_contradictions"][:10]:
        print(f"  - '{text[:60]}' labelled both {l1} and {l2}")

    print("\n--- Balance ---")
    lang_counts = {lang: dist["language"].get(lang, 0) for lang in LANGUAGES}
    print(f"Language counts      : {lang_counts} (spread={report['language_imbalance_spread']})")
    print(f"Category counts      : min={report['category_min']} max={report['category_max']}")
    print(f"Label buckets        : {dist['label_bucket']}")
    print(f"Modality             : {dist['modality']}")
    print(f"Split                : {dist['split']}")

    # A near-duplicate count is expected to be low but not necessarily zero for a
    # curated set; flag clearly when it looks excessive.
    near_ratio = report["near_duplicate_pair_count"] / max(1, report["total"])
    verdict = "OK"
    if schema_problems or report["exact_duplicate_groups"] or report["label_contradiction_count"]:
        verdict = "FAIL"
    elif near_ratio > 0.05 or report["repeated_template_count"] > 0:
        verdict = "REVIEW"
    print(f"\nOverall verdict      : {verdict}")

    if not args.no_export:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RESULTS_DIR / f"benchmark_quality_{stamp}.json"
        out_path.write_text(
            json.dumps(
                {"generated_utc": stamp, "dataset": str(dataset_path), "verdict": verdict,
                 "distribution": dist, "report": report,
                 "schema_problem_count": len(schema_problems)},
                ensure_ascii=False, indent=2, default=str,
            ),
            encoding="utf-8",
        )
        print(f"Exported             : {out_path.relative_to(ROOT)}")

    return 0 if verdict != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
