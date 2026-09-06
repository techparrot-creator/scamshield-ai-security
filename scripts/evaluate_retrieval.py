from __future__ import annotations

import json
from pathlib import Path

from scamshield.query_planner import classify_secret_flow, infer_retrieval_hints
from scamshield.retrieval import retrieve_guidance

cases = [json.loads(line) for line in (Path(__file__).resolve().parent.parent / "data" / "eval_cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
hits = 0
scored = 0
for case in cases:
    hints = infer_retrieval_hints(case["text"])
    results, confidence, _ = retrieve_guidance(case["text"], hints=hints)
    parent_ids = {item["parent_id"] for item in results}
    kind = case.get("kind", "scam")
    if kind == "legitimate":
        flow = classify_secret_flow(case["text"])
        print(f"{case['id']}: NEGATIVE CONTROL expected_risk={case.get('expected_risk')} secret_flow={flow} confidence={confidence:.3f} retrieved={sorted(parent_ids)}")
        continue
    expected = set(case.get("expected_sources", []))
    if expected:
        hit = bool(parent_ids & expected)
        hits += int(hit)
        scored += 1
        print(f"{case['id']}: {'PASS' if hit else 'MISS'} confidence={confidence:.3f} retrieved={sorted(parent_ids)}")
    else:
        print(f"{case['id']}: UNSCORED confidence={confidence:.3f} retrieved={sorted(parent_ids)}")
print(f"\nRetrieval hit@4 (scam cases with expected sources): {hits}/{scored} = {hits/scored:.1%}" if scored else "\nNo scored scam cases found.")
