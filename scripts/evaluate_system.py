"""System-level classification evaluation (separate from RAG retrieval eval).

Runs the FULL ScamShield LangGraph pipeline (Gemini assessment + advanced RAG)
over a curated/synthetic dataset (`data/system_eval_cases.jsonl`) that is
deliberately separate from the RAG knowledge base. This script makes LIVE
Gemini calls and requires GOOGLE_API_KEY in .env.

Label mapping (documented, no fabricated metrics):
- The app outputs a 4-band risk level: low / medium / high / critical.
- high, critical            -> predicted "scam"
- low                       -> predicted "legitimate"
- medium                    -> predicted "uncertain": the app is deliberately
                               calibrated to express uncertainty here, so these
                               cases are EXCLUDED from binary accuracy metrics
                               and reported separately.
- no assessment produced    -> "unavailable" (model outage): excluded and counted.
- expected == "insufficient": ambiguous-evidence cases. They are excluded from
                               binary metrics; their prediction distribution is
                               reported as a calibration check instead.

Usage:
    python scripts/evaluate_system.py
    python scripts/evaluate_system.py --limit 5
    python scripts/evaluate_system.py --dataset data/system_eval_cases.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

DEFAULT_DATASET = ROOT / "data" / "system_eval_cases.jsonl"


def predicted_class(result: dict) -> str:
    """Map a pipeline result to scam / legitimate / uncertain / unavailable."""
    assessment = result.get("assessment")
    if not assessment:
        return "unavailable"
    level = assessment.get("risk_level")
    if level in {"high", "critical"}:
        return "scam"
    if level == "low":
        return "legitimate"
    return "uncertain"


def _pct(numerator: float, denominator: float) -> str:
    return f"{numerator / denominator:.1%}" if denominator else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description="ScamShield system classification evaluation (live Gemini).")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="JSONL dataset path")
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N cases (0 = all)")
    parser.add_argument("--delay", type=float, default=0.25, help="Seconds to wait between live Gemini calls")
    args = parser.parse_args()

    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print("BLOCKED: set GOOGLE_API_KEY in .env before running the system evaluation (live Gemini calls).")
        return 2
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"BLOCKED: dataset not found: {dataset_path}")
        return 2

    from scamshield.graph import graph

    cases = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit > 0:
        cases = cases[: args.limit]

    print(f"System classification evaluation: {len(cases)} cases (live Gemini calls).")
    print("Mapping: high/critical->scam, low->legitimate, medium->uncertain (excluded), no assessment->unavailable (excluded).\n")

    runs: list[dict] = []
    for index, case in enumerate(cases):
        try:
            result = graph.invoke(
                {
                    "raw_input": case["text"],
                    "input_kind": "text",
                    "preferred_language": "English",
                }
            )
        except Exception as exc:
            result = {"assessment": None, "ai_status": f"error:{type(exc).__name__}"}
        predicted = predicted_class(result)
        assessment = result.get("assessment") or {}
        run = {
            "id": case["id"],
            "language": case.get("language", "unknown"),
            "expected": case["expected"],
            "predicted": predicted,
            "risk_level": assessment.get("risk_level", "-"),
            "risk_score": assessment.get("risk_score", "-"),
        }
        runs.append(run)
        mark = "OK" if predicted == case["expected"] else ("--" if case["expected"] == "insufficient" or predicted in {"uncertain", "unavailable"} else "XX")
        print(
            f"[{mark}] {run['id']:<24} lang={run['language']:<10} expected={run['expected']:<12} "
            f"predicted={run['predicted']:<11} risk={run['risk_level']}({run['risk_score']})"
        )
        if index < len(cases) - 1 and args.delay > 0:
            time.sleep(args.delay)

    # --- Binary metrics over cases with a definitive expected and predicted label ---
    scored = [r for r in runs if r["expected"] in {"scam", "legitimate"} and r["predicted"] in {"scam", "legitimate"}]
    tp = sum(1 for r in scored if r["expected"] == "scam" and r["predicted"] == "scam")
    fn = sum(1 for r in scored if r["expected"] == "scam" and r["predicted"] == "legitimate")
    fp = sum(1 for r in scored if r["expected"] == "legitimate" and r["predicted"] == "scam")
    tn = sum(1 for r in scored if r["expected"] == "legitimate" and r["predicted"] == "legitimate")

    uncertain = [r for r in runs if r["predicted"] == "uncertain"]
    unavailable = [r for r in runs if r["predicted"] == "unavailable"]
    insufficient = [r for r in runs if r["expected"] == "insufficient"]

    print("\n=== Classification metrics (documented mapping; uncertain/unavailable excluded) ===")
    print(f"Total cases            : {len(runs)}")
    print(f"Scored (definitive)    : {len(scored)}")
    print(f"Uncertain (medium)     : {len(uncertain)} -> excluded from binary metrics")
    print(f"Unavailable (no model) : {len(unavailable)} -> excluded from binary metrics")
    print(f"Confusion counts       : TP={tp} FN={fn} FP={fp} TN={tn}")
    print(f"Accuracy               : {_pct(tp + tn, len(scored))}")
    print(f"Scam recall            : {_pct(tp, tp + fn)}")
    print(f"Legitimate specificity : {_pct(tn, tn + fp)}")
    print(f"False-positive rate    : {_pct(fp, fp + tn)}")

    print("\n=== Language breakdown (scored cases only) ===")
    for lang in sorted({r["language"] for r in runs}):
        lang_scored = [r for r in scored if r["language"] == lang]
        correct = sum(1 for r in lang_scored if r["expected"] == r["predicted"])
        print(f"{lang:<12}: accuracy {_pct(correct, len(lang_scored))} ({correct}/{len(lang_scored)})")

    if uncertain:
        print("\n=== Uncertain predictions (app expressed calibrated uncertainty) ===")
        for r in uncertain:
            print(f"{r['id']:<24} expected={r['expected']:<12} risk={r['risk_level']}({r['risk_score']})")
    if unavailable:
        print("\n=== Unavailable assessments (model did not respond) ===")
        for r in unavailable:
            print(f"{r['id']:<24} expected={r['expected']}")
    if insufficient:
        print("\n=== Calibration check: insufficient/ambiguous expected cases ===")
        for r in insufficient:
            print(f"{r['id']:<24} predicted={r['predicted']:<11} risk={r['risk_level']}({r['risk_score']})")
        print("(These cases have no ground-truth binary label and are never counted in the metrics above.)")

    if not scored:
        print("\nWARNING: no case produced a definitive mapping; metrics are not reported.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
