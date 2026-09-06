"""Offline tests for deterministic stratified sampling and executable filtering.

These exercise the pure helpers added to :mod:`scamshield.benchmark_runner`
(``is_executable`` and ``stratified_sample``) without any live Gemini call, and
one integration check against the real deterministic benchmark build. They prove
the sampling is reproducible, reasonably balanced across language / expected
label / modality, and that executable-only selection drops media cases whose real
fixture file is missing rather than silently counting them.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from scamshield.benchmark import build_cases
from scamshield.benchmark_runner import (
    DEFAULT_SAMPLE_SEED,
    is_executable,
    select_cases,
    stratified_sample,
)


def _case(case_id, *, language="English", expected_label="scam", modality="text", fixture=None):
    case = {
        "case_id": case_id,
        "language": language,
        "expected_label": expected_label,
        "modality": modality,
    }
    if fixture is not None:
        case["fixture"] = fixture
    return case


def _symmetric_pool(per_stratum: int = 2) -> list[dict]:
    """3 languages x 3 labels x 3 modalities x ``per_stratum`` members each."""
    langs = ["English", "Roman Urdu", "Urdu"]
    labels = ["scam", "legitimate", "ambiguous"]
    mods = ["text", "email_link", "screenshot"]
    pool: list[dict] = []
    n = 0
    for lang in langs:
        for label in labels:
            for mod in mods:
                for _ in range(per_stratum):
                    n += 1
                    pool.append(_case(f"c{n:03d}", language=lang, expected_label=label, modality=mod))
    return pool


def _spread(cases: list[dict], field: str) -> int:
    counts = Counter(c[field] for c in cases)
    return max(counts.values()) - min(counts.values())


# --------------------------------------------------------------------------- #
# is_executable
# --------------------------------------------------------------------------- #
def test_is_executable_true_for_text_and_email():
    root = Path(".")
    assert is_executable(_case("t", modality="text"), root) is True
    assert is_executable(_case("e", modality="email_link"), root) is True


def test_is_executable_false_for_media_without_file(tmp_path):
    case = _case("s", modality="screenshot", fixture={"path": "missing.png", "kind": "image"})
    assert is_executable(case, tmp_path) is False


def test_is_executable_false_when_media_has_no_fixture_metadata(tmp_path):
    assert is_executable(_case("a", modality="audio"), tmp_path) is False


def test_is_executable_true_when_media_fixture_exists(tmp_path):
    (tmp_path / "shot.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    case = _case("s", modality="screenshot", fixture={"path": "shot.png", "kind": "image"})
    assert is_executable(case, tmp_path) is True


def test_executable_only_filter_drops_fixtureless_media(tmp_path):
    (tmp_path / "ok.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    pool = [
        _case("text1", modality="text"),
        _case("email1", modality="email_link"),
        _case("shot_ok", modality="screenshot", fixture={"path": "ok.png", "kind": "image"}),
        _case("shot_missing", modality="screenshot", fixture={"path": "gone.png", "kind": "image"}),
        _case("audio_missing", modality="audio", fixture={"path": "gone.wav", "kind": "audio"}),
    ]
    kept = [c["case_id"] for c in pool if is_executable(c, tmp_path)]
    assert kept == ["text1", "email1", "shot_ok"]


# --------------------------------------------------------------------------- #
# stratified_sample: size, determinism, no duplicates
# --------------------------------------------------------------------------- #
def test_stratified_sample_respects_size():
    assert len(stratified_sample(_symmetric_pool(3), 30)) == 30


def test_stratified_sample_no_duplicates():
    sample = stratified_sample(_symmetric_pool(3), 30)
    ids = [c["case_id"] for c in sample]
    assert len(ids) == len(set(ids)) == 30


def test_stratified_sample_is_deterministic_for_seed():
    pool = _symmetric_pool(3)
    a = [c["case_id"] for c in stratified_sample(pool, 30, seed=42)]
    b = [c["case_id"] for c in stratified_sample(pool, 30, seed=42)]
    assert a == b


def test_stratified_sample_default_seed_is_stable():
    pool = _symmetric_pool(2)
    a = [c["case_id"] for c in stratified_sample(pool, 20)]
    b = [c["case_id"] for c in stratified_sample(pool, 20, seed=DEFAULT_SAMPLE_SEED)]
    assert a == b


def test_stratified_sample_independent_of_input_order():
    pool = _symmetric_pool(2)
    a = [c["case_id"] for c in stratified_sample(pool, 20, seed=7)]
    b = [c["case_id"] for c in stratified_sample(list(reversed(pool)), 20, seed=7)]
    assert a == b


def test_stratified_sample_different_seed_can_differ():
    pool = _symmetric_pool(4)  # 108 cases; plenty to draw 30 differently per seed
    a = [c["case_id"] for c in stratified_sample(pool, 30, seed=1)]
    b = [c["case_id"] for c in stratified_sample(pool, 30, seed=2)]
    assert a != b


# --------------------------------------------------------------------------- #
# stratified_sample: full-pool passthrough preserves existing behavior
# --------------------------------------------------------------------------- #
def test_stratified_sample_size_zero_returns_full_pool():
    pool = _symmetric_pool(1)
    assert len(stratified_sample(pool, 0)) == len(pool)


def test_stratified_sample_size_ge_pool_returns_full_pool():
    pool = _symmetric_pool(1)  # 27
    sample = stratified_sample(pool, 100)
    assert len(sample) == len(pool)
    assert {c["case_id"] for c in sample} == {c["case_id"] for c in pool}


# --------------------------------------------------------------------------- #
# stratified_sample: balance across the three marginals
# --------------------------------------------------------------------------- #
def test_stratified_sample_balances_language():
    sample = stratified_sample(_symmetric_pool(3), 30)
    counts = Counter(c["language"] for c in sample)
    assert set(counts) == {"English", "Roman Urdu", "Urdu"}, dict(counts)
    assert max(counts.values()) - min(counts.values()) <= 1, dict(counts)


def test_stratified_sample_balances_label():
    sample = stratified_sample(_symmetric_pool(3), 30)
    counts = Counter(c["expected_label"] for c in sample)
    assert set(counts) == {"scam", "legitimate", "ambiguous"}, dict(counts)
    assert max(counts.values()) - min(counts.values()) <= 2, dict(counts)


def test_stratified_sample_balances_modality_reasonably():
    sample = stratified_sample(_symmetric_pool(3), 30)
    counts = Counter(c["modality"] for c in sample)
    assert set(counts) == {"text", "email_link", "screenshot"}, dict(counts)
    assert max(counts.values()) - min(counts.values()) <= 2, dict(counts)


def test_stratified_sample_handles_sparse_strata():
    pool = [
        _case("a1", language="English", expected_label="scam", modality="text"),
        _case("a2", language="English", expected_label="scam", modality="text"),
        _case("b1", language="Urdu", expected_label="scam", modality="text"),
    ]
    sample = stratified_sample(pool, 2)
    assert len(sample) == 2
    # Language is the top priority, so the two picks span both languages.
    assert {c["language"] for c in sample} == {"English", "Urdu"}


def test_stratified_sample_stops_when_pool_exhausted():
    pool = _symmetric_pool(1)  # 27 cases
    sample = stratified_sample(pool, 27)
    assert len(sample) == 27


# --------------------------------------------------------------------------- #
# Integration against the real deterministic benchmark build
# --------------------------------------------------------------------------- #
def test_stratified_sample_on_real_holdout_is_balanced_and_executable(tmp_path):
    cases = select_cases(build_cases(), "holdout")
    # tmp_path holds no media fixtures, so executable-only keeps text/email cases.
    executable = [c for c in cases if is_executable(c, tmp_path)]
    assert executable
    assert all(c["modality"] not in {"screenshot", "audio"} for c in executable)

    sample = stratified_sample(executable, 30)
    assert len(sample) == 30
    ids = [c["case_id"] for c in sample]
    assert len(set(ids)) == 30

    lang_counts = Counter(c["language"] for c in sample)
    assert set(lang_counts) == {"English", "Roman Urdu", "Urdu"}, dict(lang_counts)
    assert max(lang_counts.values()) - min(lang_counts.values()) <= 2, dict(lang_counts)

    label_counts = Counter(c["expected_label"] for c in sample)
    # scam + legitimate must both appear; ambiguous appears where the pool allows.
    assert {"scam", "legitimate"} <= set(label_counts), dict(label_counts)
