"""Tests for the multilingual evaluation benchmark and its evaluator metrics.

These are OFFLINE tests: they build the benchmark deterministically (no Gemini,
no network, no API key) and validate schema, distribution, split isolation,
adversarial-pair integrity, dataset quality, and the pure metric helpers.
"""
from __future__ import annotations

import pytest

from scamshield.benchmark import (
    LABELS,
    LANGUAGES,
    MEDIA_KIND,
    MODALITIES,
    SPLITS,
    TARGET_LABEL_BUCKETS,
    TARGET_LANGUAGES,
    TARGET_MODALITIES,
    TOTAL_CASES,
    build_cases,
    case_evidence_text,
    classification_metrics,
    confusion_counts,
    distributions,
    exact_duplicates,
    label_bucket,
    label_contradictions,
    near_duplicate_pairs,
    predicted_class,
    repeated_templates,
    summarize_runs,
    validate_case,
)


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    return build_cases()


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #
def test_every_case_passes_schema_validation(cases):
    problems = [(c["case_id"], p) for c in cases for p in validate_case(c)]
    assert problems == []


def test_case_count_is_exactly_600(cases):
    assert len(cases) == TOTAL_CASES == 600


def test_enum_fields_are_within_schema(cases):
    for case in cases:
        assert case["language"] in LANGUAGES
        assert case["modality"] in MODALITIES
        assert case["expected_label"] in LABELS
        assert case["split"] in SPLITS


# --------------------------------------------------------------------------- #
# No duplicate case IDs
# --------------------------------------------------------------------------- #
def test_case_ids_are_unique(cases):
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------- #
# Class / language / modality distribution
# --------------------------------------------------------------------------- #
def test_label_bucket_distribution_matches_targets(cases):
    dist = distributions(cases)
    assert dist["label_bucket"] == TARGET_LABEL_BUCKETS
    assert sum(TARGET_LABEL_BUCKETS.values()) == TOTAL_CASES


def test_language_distribution_is_balanced(cases):
    dist = distributions(cases)
    assert dist["language"] == TARGET_LANGUAGES


def test_modality_distribution_matches_targets(cases):
    dist = distributions(cases)
    assert dist["modality"] == TARGET_MODALITIES
    assert sum(TARGET_MODALITIES.values()) == TOTAL_CASES


def test_label_bucket_helper_folds_adversarial(cases):
    adversarial = [c for c in cases if c["source_type"] == "adversarial_pair"]
    assert adversarial, "expected adversarial cases to exist"
    assert all(label_bucket(c) == "adversarial" for c in adversarial)
    # Adversarial members still carry a real scam/legitimate label for scoring.
    assert {c["expected_label"] for c in adversarial} == {"scam", "legitimate"}


# --------------------------------------------------------------------------- #
# Deterministic split + holdout isolation
# --------------------------------------------------------------------------- #
def test_build_is_deterministic(cases):
    assert build_cases() == cases


def test_split_sizes_are_roughly_400_200(cases):
    dist = distributions(cases)
    dev, holdout = dist["split"]["development"], dist["split"]["holdout"]
    assert dev + holdout == TOTAL_CASES
    assert 380 <= dev <= 420
    assert 180 <= holdout <= 220


def test_holdout_is_isolated_from_development(cases):
    dev_ids = {c["case_id"] for c in cases if c["split"] == "development"}
    holdout_ids = {c["case_id"] for c in cases if c["split"] == "holdout"}
    assert not (dev_ids & holdout_ids)
    assert dev_ids and holdout_ids


def test_split_assignment_is_reproducible(cases):
    again = {c["case_id"]: c["split"] for c in build_cases()}
    assert {c["case_id"]: c["split"] for c in cases} == again


# --------------------------------------------------------------------------- #
# Adversarial pair integrity
# --------------------------------------------------------------------------- #
def test_adversarial_pairs_are_complete_and_consistent(cases):
    pairs: dict[str, list[dict]] = {}
    for case in cases:
        if case["source_type"] == "adversarial_pair":
            pairs.setdefault(case["adversarial_pair_id"], []).append(case)
    assert len(pairs) == 30  # 60 adversarial cases / 2 members
    for pair_id, members in pairs.items():
        assert len(members) == 2, pair_id
        labels = sorted(m["expected_label"] for m in members)
        assert labels == ["legitimate", "scam"], pair_id
        assert len({m["language"] for m in members}) == 1, pair_id
        assert len({m["category"] for m in members}) == 1, pair_id
        assert all(m["difficulty"] == "hard" for m in members)
        # Same split: a pair is never torn across development/holdout.
        assert len({m["split"] for m in members}) == 1, pair_id
        # The two members must not be near-identical text (intent differs, not one word).
        scam = next(m for m in members if m["expected_label"] == "scam")
        legit = next(m for m in members if m["expected_label"] == "legitimate")
        assert case_evidence_text(scam) != case_evidence_text(legit)


# --------------------------------------------------------------------------- #
# Media cases carry fixture metadata; no fabricated multimodal content
# --------------------------------------------------------------------------- #
def test_media_cases_reference_fixtures(cases):
    media = [c for c in cases if c["modality"] in MEDIA_KIND]
    assert media
    for case in media:
        fixture = case["fixture"]
        assert fixture["status"] == "requires_real_fixture"
        assert fixture["kind"] == MEDIA_KIND[case["modality"]]
        assert fixture["render_text"].strip()
        assert case["text_or_fixture_reference"] == fixture["path"]


# --------------------------------------------------------------------------- #
# Dataset quality
# --------------------------------------------------------------------------- #
def test_no_exact_duplicates_or_label_contradictions(cases):
    assert exact_duplicates(cases) == {}
    assert label_contradictions(cases) == []


def test_no_repeated_templates(cases):
    assert repeated_templates(cases) == {}


def test_near_duplicates_are_not_excessive(cases):
    near = near_duplicate_pairs(cases, threshold=0.94)
    assert len(near) / len(cases) <= 0.05


# --------------------------------------------------------------------------- #
# Benchmark is separate from the RAG knowledge base (no leakage)
# --------------------------------------------------------------------------- #
def test_benchmark_not_embedded_in_rag_knowledge(cases):
    import pathlib

    repo_knowledge = pathlib.Path(__file__).resolve().parent.parent / "knowledge" / "sources.json"
    if not repo_knowledge.exists():
        pytest.skip("knowledge/sources.json not present")
    blob = repo_knowledge.read_text(encoding="utf-8")
    sample_texts = [case_evidence_text(c) for c in cases]
    # No benchmark evidence text may appear verbatim in the RAG corpus (no leakage).
    assert not any(text and text in blob for text in sample_texts)


# --------------------------------------------------------------------------- #
# Evaluator metric correctness (pure helpers)
# --------------------------------------------------------------------------- #
def _run(expected, predicted, **extra):
    base = {
        "expected_label": expected, "predicted": predicted,
        "language": "English", "category": "c", "modality": "text", "difficulty": "easy",
    }
    base.update(extra)
    return base


def test_predicted_class_mapping():
    assert predicted_class({"assessment": {"risk_level": "high"}}) == "scam"
    assert predicted_class({"assessment": {"risk_level": "critical"}}) == "scam"
    assert predicted_class({"assessment": {"risk_level": "low"}}) == "legitimate"
    assert predicted_class({"assessment": {"risk_level": "medium"}}) == "uncertain"
    assert predicted_class({"assessment": None}) == "unavailable"
    assert predicted_class({}) == "unavailable"


def test_confusion_counts_exclude_ambiguous_and_abstentions():
    runs = [
        _run("scam", "scam"), _run("scam", "scam"),        # tp = 2
        _run("scam", "legitimate"),                         # fn = 1
        _run("legitimate", "legitimate"), _run("legitimate", "legitimate"), _run("legitimate", "legitimate"),  # tn = 3
        _run("legitimate", "scam"),                         # fp = 1
        _run("scam", "uncertain"),                          # excluded (abstention)
        _run("legitimate", "unavailable"),                  # excluded (no model)
        _run("ambiguous", "scam"),                          # excluded (no ground truth)
        _run("ambiguous", "uncertain"),                     # excluded
    ]
    counts = confusion_counts(runs)
    assert counts == {"tp": 2, "fn": 1, "fp": 1, "tn": 3, "scored": 7}


def test_classification_metrics_values():
    counts = {"tp": 8, "tn": 9, "fp": 1, "fn": 2, "scored": 20}
    metrics = classification_metrics(counts)
    assert metrics["accuracy"] == pytest.approx(17 / 20)
    assert metrics["scam_recall"] == pytest.approx(8 / 10)
    assert metrics["legitimate_specificity"] == pytest.approx(9 / 10)
    assert metrics["false_positive_rate"] == pytest.approx(1 / 10)
    assert metrics["precision"] == pytest.approx(8 / 9)
    expected_f1 = 2 * (8 / 9) * (8 / 10) / ((8 / 9) + (8 / 10))
    assert metrics["f1"] == pytest.approx(expected_f1)


def test_classification_metrics_handles_empty_and_zero_division():
    empty = classification_metrics({"tp": 0, "tn": 0, "fp": 0, "fn": 0, "scored": 0})
    assert empty["accuracy"] is None
    assert empty["scam_recall"] is None
    assert empty["precision"] is None
    # No false positives and no true negatives -> specificity undefined.
    undefined = classification_metrics({"tp": 5, "tn": 0, "fp": 0, "fn": 0, "scored": 5})
    assert undefined["legitimate_specificity"] is None
    assert undefined["scam_recall"] == pytest.approx(1.0)


def test_summarize_runs_totals_and_abstention_rate():
    runs = [
        _run("scam", "scam"), _run("scam", "legitimate"),
        _run("legitimate", "legitimate"), _run("legitimate", "scam"),
        _run("scam", "uncertain"), _run("legitimate", "unavailable"),
        _run("ambiguous", "uncertain"),
    ]
    summary = summarize_runs(runs)
    assert summary["total"] == 7
    assert summary["definitive"] == 4
    assert summary["ambiguous_expected"] == 1
    assert summary["predicted_uncertain"] == 2
    assert summary["predicted_unavailable"] == 1
    # No missing fixtures here: everything executed, one real model failure.
    assert summary["fixture_unavailable"] == 0
    assert summary["executed"] == 7
    assert summary["execution_coverage"] == pytest.approx(1.0)
    assert summary["model_responded"] == 6
    # Abstention counts ONLY real model uncertainty, over cases the model answered.
    assert summary["abstention_rate"] == pytest.approx(2 / 6)
    assert summary["counts"] == {"tp": 1, "fn": 1, "fp": 1, "tn": 1, "scored": 4}
    assert set(summary["breakdowns"]) == {"language", "category", "modality", "difficulty"}


def test_summarize_runs_counts_fixture_unavailable():
    runs = [_run("scam", "unavailable", status="fixture_unavailable"), _run("scam", "scam", status="executed")]
    summary = summarize_runs(runs)
    assert summary["fixture_unavailable"] == 1
    # A missing fixture is NOT a model failure and NOT model uncertainty.
    assert summary["predicted_unavailable"] == 0
    assert summary["predicted_uncertain"] == 0
    assert summary["executed"] == 1
    assert summary["execution_coverage"] == pytest.approx(0.5)
    assert summary["model_responded"] == 1
    assert summary["abstention_rate"] == pytest.approx(0.0)


def test_breakdown_groups_by_language():
    runs = [
        _run("scam", "scam", language="English"),
        _run("legitimate", "legitimate", language="Urdu"),
        _run("legitimate", "scam", language="Urdu"),
    ]
    summary = summarize_runs(runs)
    assert summary["breakdowns"]["language"]["English"]["tp"] == 1
    urdu = summary["breakdowns"]["language"]["Urdu"]
    assert urdu["fp"] == 1 and urdu["tn"] == 1 and urdu["scored"] == 2


# --------------------------------------------------------------------------- #
# Committed benchmark file (if present) matches the deterministic build
# --------------------------------------------------------------------------- #
def test_committed_benchmark_matches_build():
    import pathlib

    path = pathlib.Path(__file__).resolve().parent.parent / "data" / "evaluation" / "scamshield_benchmark.jsonl"
    if not path.exists():
        pytest.skip("benchmark JSONL not built yet; run scripts/build_benchmark.py")
    from scamshield.benchmark import load_jsonl

    on_disk = load_jsonl(path)
    assert len(on_disk) == TOTAL_CASES
    assert {c["case_id"] for c in on_disk} == {c["case_id"] for c in build_cases()}
