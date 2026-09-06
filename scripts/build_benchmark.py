"""Build the ScamShield multilingual evaluation benchmark.

Emits a deterministic ~600-case benchmark that is kept SEPARATE from the RAG
knowledge base in ``knowledge/sources.json``. Benchmark cases are evaluation
fixtures only and are never indexed for retrieval (avoids evaluation leakage).

Outputs:
    data/evaluation/scamshield_benchmark.jsonl            (all 600 cases)
    data/evaluation/scamshield_benchmark_dev.jsonl        (development/calibration split)
    data/evaluation/scamshield_benchmark_holdout.jsonl    (LOCKED holdout split)

The holdout split must NOT be used for threshold tuning; it is reserved for the
baseline/ablation comparison and final reporting.

Usage:
    python scripts/build_benchmark.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scamshield.benchmark import (  # noqa: E402
    TOTAL_CASES,
    TARGET_LABEL_BUCKETS,
    TARGET_LANGUAGES,
    TARGET_MODALITIES,
    build_cases,
    distributions,
    validate_case,
    write_jsonl,
)

OUT_DIR = ROOT / "data" / "evaluation"
FULL_PATH = OUT_DIR / "scamshield_benchmark.jsonl"
DEV_PATH = OUT_DIR / "scamshield_benchmark_dev.jsonl"
HOLDOUT_PATH = OUT_DIR / "scamshield_benchmark_holdout.jsonl"


def _print_distribution(dist: dict) -> None:
    print(f"Total cases : {dist['total']}")
    print("\nLabel buckets (mutually exclusive):")
    for key in ("scam", "legitimate", "ambiguous", "adversarial"):
        print(f"  {key:<12}: {dist['label_bucket'].get(key, 0)}")
    print("\nUnderlying expected_label (adversarial folded back in):")
    for key in ("scam", "legitimate", "ambiguous"):
        print(f"  {key:<12}: {dist['expected_label'].get(key, 0)}")
    print("\nLanguages:")
    for key in ("English", "Roman Urdu", "Urdu"):
        print(f"  {key:<12}: {dist['language'].get(key, 0)}")
    print("\nModalities:")
    for key in ("text", "email_link", "screenshot", "audio"):
        print(f"  {key:<12}: {dist['modality'].get(key, 0)}")
    print("\nDifficulty:")
    for key in ("easy", "medium", "hard"):
        print(f"  {key:<12}: {dist['difficulty'].get(key, 0)}")
    print("\nSplits:")
    for key in ("development", "holdout"):
        print(f"  {key:<12}: {dist['split'].get(key, 0)}")
    print(f"\nDistinct categories: {len(dist['category'])}")


def main() -> int:
    cases = build_cases()

    # Hard schema gate: refuse to write an invalid benchmark.
    problems: list[tuple[str, str]] = []
    for case in cases:
        for problem in validate_case(case):
            problems.append((case["case_id"], problem))
    if problems:
        print(f"BLOCKED: {len(problems)} schema problem(s) detected; benchmark not written.")
        for case_id, problem in problems[:20]:
            print(f"  - {case_id}: {problem}")
        return 1

    dist = distributions(cases)

    # Assert the promised distribution so a regression cannot silently ship.
    assert dist["total"] == TOTAL_CASES, dist["total"]
    for key, target in TARGET_LABEL_BUCKETS.items():
        assert dist["label_bucket"].get(key, 0) == target, (key, dist["label_bucket"])
    for key, target in TARGET_LANGUAGES.items():
        assert dist["language"].get(key, 0) == target, (key, dist["language"])
    for key, target in TARGET_MODALITIES.items():
        assert dist["modality"].get(key, 0) == target, (key, dist["modality"])

    dev = [c for c in cases if c["split"] == "development"]
    holdout = [c for c in cases if c["split"] == "holdout"]
    assert len(dev) + len(holdout) == len(cases)
    assert not ({c["case_id"] for c in dev} & {c["case_id"] for c in holdout})

    write_jsonl(cases, FULL_PATH)
    write_jsonl(dev, DEV_PATH)
    write_jsonl(holdout, HOLDOUT_PATH)

    print("ScamShield benchmark built and validated (deterministic, synthetic/curated).")
    _print_distribution(dist)
    print(f"\nWrote:\n  {FULL_PATH.relative_to(ROOT)}\n  {DEV_PATH.relative_to(ROOT)}\n  {HOLDOUT_PATH.relative_to(ROOT)}")
    print(
        "\nNote: benchmark cases are evaluation fixtures only. They are NOT added to "
        "knowledge/sources.json and are never indexed into ChromaDB/BM25 (no leakage)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
