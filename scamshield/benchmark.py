"""ScamShield multilingual evaluation benchmark: schema, generation, split, and metrics.

This module is the single source of truth for the hackathon-grade benchmark. It
is intentionally decoupled from the RAG corpus in ``knowledge/sources.json``:
benchmark cases are evaluation fixtures and are NEVER indexed for retrieval.

Design guarantees (all deterministic, no RNG unless seeded):
- Exact label buckets: 240 scam, 210 legitimate, 90 ambiguous, 60 adversarial (=600).
- Exact language totals: 200 English, 200 Roman Urdu, 200 Urdu.
- Exact modality totals: 360 text, 90 email/link, 90 screenshot, 60 audio.
- A deterministic stratified split into ``development`` (~400) and ``holdout`` (~200).
- Adversarial scam/legitimate members of a pair always land in the same split.

The pure metric helpers here (``confusion_counts``, ``classification_metrics``,
``summarize_runs``) are exercised by unit tests without any live Gemini calls.
"""
from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from scamshield.benchmark_content import (
    ADVERSARIAL_PAIRS,
    AMBIGUOUS_CATEGORIES,
    LEGITIMATE_CATEGORIES,
    SCAM_CATEGORIES,
)

# --------------------------------------------------------------------------- #
# Schema constants
# --------------------------------------------------------------------------- #
LANGUAGES = ("English", "Roman Urdu", "Urdu")
LANGUAGE_KEYS = {"English": "en", "Roman Urdu": "roman", "Urdu": "urdu"}
MODALITIES = ("text", "email_link", "screenshot", "audio")
LABELS = ("scam", "legitimate", "ambiguous")
RISK_BANDS = ("low", "medium", "high", "critical")
DIFFICULTIES = ("easy", "medium", "hard")
SPLITS = ("development", "holdout")
SOURCE_TYPES = ("curated_scam", "curated_legitimate", "ambiguous", "adversarial_pair")

REQUIRED_FIELDS = (
    "case_id",
    "language",
    "modality",
    "category",
    "expected_label",
    "expected_risk_band",
    "text_or_fixture_reference",
    "expected_signals",
    "difficulty",
    "source_type",
    "notes",
    "split",
)

# Target distribution (mutually exclusive label buckets sum to 600).
TARGET_LABEL_BUCKETS = {"scam": 240, "legitimate": 210, "ambiguous": 90, "adversarial": 60}
TARGET_LANGUAGES = {"English": 200, "Roman Urdu": 200, "Urdu": 200}
TARGET_MODALITIES = {"text": 360, "email_link": 90, "screenshot": 90, "audio": 60}
TOTAL_CASES = 600
BUILD_SEED = 20260904
SPLIT_SEED = 730913

# --------------------------------------------------------------------------- #
# Deterministic slot filling (synthetic, privacy-safe values only)
# --------------------------------------------------------------------------- #
AMOUNTS = ("PKR 15,000", "PKR 25,000", "PKR 40,000", "PKR 8,500", "PKR 60,000", "PKR 12,000")
FEES = ("PKR 2,500", "PKR 1,500", "PKR 3,000", "PKR 999", "PKR 5,000", "PKR 1,200")
CODES = ("483211", "558899", "120934", "774210", "309158", "612407")
PLATFORMS = ("JazzCash", "Easypaisa", "EasyPaisa wallet", "SadaPay", "NayaPay")
DEADLINES = ("2 hours", "24 hours", "today", "30 minutes", "12 hours", "tonight")
BANKS = ("Meezan Bank", "HBL", "UBL", "Allied Bank", "Bank Alfalah", "Askari Bank")
SCAM_URLS = (
    "http://bit.ly/secure-verify-now",
    "http://account-update-portal.tk/login",
    "https://parcel-clearance-pay.top/fee",
    "http://bank-secure-login.xyz/confirm",
)
NEUTRAL_URLS = ("https://example-form.app/signin", "http://link-shared-to-me.app/page")
PHONES = ("+92-300-000-0111", "+92-321-000-0222", "+92-333-000-0333")


def fill_slots(text: str, rng: random.Random, *, suspicious_link: bool) -> str:
    """Replace slot tokens with deterministic synthetic values."""
    url_pool = SCAM_URLS if suspicious_link else NEUTRAL_URLS
    mapping = {
        "{amount}": rng.choice(AMOUNTS),
        "{amount2}": rng.choice(FEES),
        "{code}": rng.choice(CODES),
        "{platform}": rng.choice(PLATFORMS),
        "{deadline}": rng.choice(DEADLINES),
        "{bank}": rng.choice(BANKS),
        "{url}": rng.choice(url_pool),
        "{phone}": rng.choice(PHONES),
    }
    for token, value in mapping.items():
        text = text.replace(token, value)
    return text


# --------------------------------------------------------------------------- #
# Modality affinity (keeps channel assignment plausible while hitting totals)
# --------------------------------------------------------------------------- #
MODALITY_AFFINITY: dict[str, dict[str, int]] = {
    "otp_pin_theft": {"screenshot": 2, "audio": 1},
    "phishing_smishing": {"email_link": 3},
    "bank_impersonation": {"email_link": 2, "audio": 1},
    "whatsapp_takeover": {"screenshot": 3},
    "fake_job_task": {"screenshot": 2, "email_link": 1},
    "fake_seller_buyer": {"screenshot": 3},
    "investment_crypto": {"email_link": 2},
    "courier_impersonation": {"email_link": 3},
    "government_impersonation": {"audio": 2, "email_link": 1},
    "remote_access": {"audio": 2},
    "lottery_prize": {"email_link": 2, "screenshot": 1},
    "romance": {"screenshot": 2, "audio": 2},
    "blackmail_extortion": {"audio": 3, "screenshot": 1},
    "payment_recovery": {"email_link": 1, "audio": 1},
    "account_recovery": {"audio": 1, "email_link": 1},
    "legit_otp_notice": {"screenshot": 2},
    "legit_transaction_alert": {"screenshot": 2, "email_link": 1},
    "legit_password_reset": {"email_link": 3},
    "legit_courier_update": {"email_link": 2},
    "legit_job_offer": {"email_link": 3},
    "legit_marketplace": {"screenshot": 3},
    "legit_security_notification": {"email_link": 3},
    "legit_support_communication": {"email_link": 2, "audio": 1},
    "legit_personal_chat": {"screenshot": 3, "audio": 2},
    "legit_payment_receipt": {"email_link": 2, "screenshot": 1},
    "ambiguous_vague_money": {"audio": 2, "screenshot": 1},
    "ambiguous_confirm_details": {"email_link": 2},
    "ambiguous_unverified_link": {"email_link": 2, "screenshot": 2},
    "ambiguous_personal_request": {"audio": 2, "screenshot": 1},
    # adversarial categories
    "otp_direction": {"screenshot": 3},
    "google_notice": {"email_link": 3},
    "job_offer": {"email_link": 3},
    "transaction_alert": {"screenshot": 2},
    "delivery_fee": {"email_link": 3},
    "support_call": {"audio": 3},
    "investment_pitch": {"email_link": 3},
    "marketplace_qr": {"screenshot": 3},
    "account_security": {"email_link": 3},
    "prize_claim": {"email_link": 3},
}

CRITICAL_CATEGORIES = {"blackmail_extortion", "romance", "government_impersonation"}

MEDIA_KIND = {"screenshot": "image", "audio": "audio"}
FIXTURE_EXT = {"screenshot": ".png", "audio": ".wav"}


def _risk_band(label: str, category: str) -> str:
    if label == "legitimate":
        return "low"
    if label == "ambiguous":
        return "medium"
    return "critical" if category in CRITICAL_CATEGORIES else "high"


def _fixture(case_id: str, modality: str, message: str) -> dict:
    """Build fixture metadata for screenshot/audio cases.

    The project has no supported offline synthetic image/audio generation path
    (media extraction requires live Gemini over real bytes), so these cases ship
    with metadata + the intended render text and are marked ``requires_real_fixture``.
    They are NEVER counted as an executed multimodal PASS.
    """
    subdir = "screenshots" if modality == "screenshot" else "audio"
    rel_path = f"data/evaluation/fixtures/{subdir}/{case_id}{FIXTURE_EXT[modality]}"
    return {
        "kind": MEDIA_KIND[modality],
        "path": rel_path,
        "status": "requires_real_fixture",
        "render_text": message,
        "how_to_provide": (
            "Create a real screenshot/voice-note fixture containing render_text, save it at "
            "'path', then the evaluator runs the live multimodal path via scamshield.media."
        ),
    }


def _new_case(
    *,
    case_id: str,
    language: str,
    category: str,
    label: str,
    source_type: str,
    signals: list[str],
    difficulty: str,
    notes: str,
    message: str,
    adversarial_pair_id: str | None = None,
) -> dict:
    case = {
        "case_id": case_id,
        "language": language,
        "modality": "text",  # assigned later to hit exact modality totals
        "category": category,
        "expected_label": label,
        "expected_risk_band": _risk_band(label, category),
        "text_or_fixture_reference": message,
        "expected_signals": signals,
        "difficulty": difficulty,
        "source_type": source_type,
        "notes": notes,
        "split": "development",  # assigned later
    }
    if adversarial_pair_id:
        case["adversarial_pair_id"] = adversarial_pair_id
    return case


def _distribute(total: int, cells: int) -> list[int]:
    """Split ``total`` into ``cells`` near-equal parts, remainder to earliest cells."""
    base, rem = divmod(total, cells)
    return [base + (1 if i < rem else 0) for i in range(cells)]


def _build_bucket_cases(bucket: str, categories: dict, per_language: int, source_type: str, label: str) -> list[dict]:
    cases: list[dict] = []
    cat_names = sorted(categories)
    for language in LANGUAGES:
        lang_key = LANGUAGE_KEYS[language]
        per_cat = _distribute(per_language, len(cat_names))
        for category, count in zip(cat_names, per_cat):
            meta = categories[category]
            templates = meta[lang_key][:count]
            for index, template in enumerate(templates, start=1):
                case_id = f"{bucket}-{category}-{lang_key}-{index:02d}"
                rng = random.Random(f"{BUILD_SEED}:{case_id}")
                message = fill_slots(template, rng, suspicious_link=(label == "scam"))
                cases.append(
                    _new_case(
                        case_id=case_id,
                        language=language,
                        category=category,
                        label=label,
                        source_type=source_type,
                        signals=list(meta.get("signals", [])),
                        difficulty=meta.get("difficulty", "medium"),
                        notes=f"Curated synthetic {label} example for category '{category}'.",
                        message=message,
                    )
                )
    return cases


def _build_adversarial_cases() -> list[dict]:
    cases: list[dict] = []
    for pair_index, pair in enumerate(ADVERSARIAL_PAIRS, start=1):
        category = pair["category"]
        for language in LANGUAGES:
            lang_key = LANGUAGE_KEYS[language]
            pair_id = f"adv-{category}-{lang_key}"
            texts = pair[lang_key]
            for member, label in (("scam", "scam"), ("legit", "legitimate")):
                rng = random.Random(f"{BUILD_SEED}:{pair_id}:{member}")
                message = fill_slots(texts[member], rng, suspicious_link=(label == "scam"))
                signals = pair["signals_scam"] if member == "scam" else pair["signals_legit"]
                case_id = f"adv-{category}-{lang_key}-{member}"
                cases.append(
                    _new_case(
                        case_id=case_id,
                        language=language,
                        category=category,
                        label=label,
                        source_type="adversarial_pair",
                        signals=list(signals),
                        difficulty="hard",
                        notes=(
                            f"Adversarial pair '{category}' ({member} member); differs from its "
                            "counterpart by critical intent, not a single swapped word."
                        ),
                        message=message,
                        adversarial_pair_id=pair_id,
                    )
                )
    return cases


def _assign_modalities(cases: list[dict]) -> None:
    """Assign modalities in place, hitting TARGET_MODALITIES exactly via affinity ranking."""
    remaining = list(range(len(cases)))
    for modality, target in (("audio", TARGET_MODALITIES["audio"]),
                             ("screenshot", TARGET_MODALITIES["screenshot"]),
                             ("email_link", TARGET_MODALITIES["email_link"])):
        ranked = sorted(
            remaining,
            key=lambda i: (-MODALITY_AFFINITY.get(cases[i]["category"], {}).get(modality, 0), i),
        )
        chosen = ranked[:target]
        for i in chosen:
            cases[i]["modality"] = modality
        remaining = [i for i in remaining if i not in set(chosen)]
    for i in remaining:  # everything else is plain text
        cases[i]["modality"] = "text"


def _apply_modality_reference(cases: list[dict]) -> None:
    """For screenshot/audio cases, swap the inline text for a fixture reference."""
    for case in cases:
        modality = case["modality"]
        if modality in MEDIA_KIND:
            message = case["text_or_fixture_reference"]
            fixture = _fixture(case["case_id"], modality, message)
            case["fixture"] = fixture
            case["text_or_fixture_reference"] = fixture["path"]
            case["notes"] = (
                case["notes"]
                + f" Delivered as a {modality}; requires a real fixture file at '{fixture['path']}'."
            )


def _assign_splits(cases: list[dict]) -> None:
    """Deterministic stratified split (~1/3 holdout). Adversarial pairs stay together."""
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for case in cases:
        bucket = "adversarial" if case["source_type"] == "adversarial_pair" else case["expected_label"]
        strata[(bucket, case["language"])].append(case)

    rng = random.Random(SPLIT_SEED)
    for key in sorted(strata):
        members = strata[key]
        # A split unit is a case, except adversarial pairs which move together.
        units: dict[str, list[dict]] = defaultdict(list)
        for case in members:
            unit_key = case.get("adversarial_pair_id", case["case_id"])
            units[unit_key].append(case)
        unit_ids = list(units)  # deterministic insertion order
        shuffled = rng.sample(unit_ids, len(unit_ids))
        cases_per_unit = len(members) // len(unit_ids)
        target_holdout_units = max(0, round(len(members) / 3 / cases_per_unit))
        holdout_units = set(shuffled[:target_holdout_units])
        for unit_id, unit_cases in units.items():
            split = "holdout" if unit_id in holdout_units else "development"
            for case in unit_cases:
                case["split"] = split


def build_cases() -> list[dict]:
    """Build the full deterministic 600-case benchmark."""
    cases: list[dict] = []
    cases += _build_bucket_cases("scam", SCAM_CATEGORIES, 80, "curated_scam", "scam")
    cases += _build_bucket_cases("legit", LEGITIMATE_CATEGORIES, 70, "curated_legitimate", "legitimate")
    cases += _build_bucket_cases("ambig", AMBIGUOUS_CATEGORIES, 30, "ambiguous", "ambiguous")
    cases += _build_adversarial_cases()

    _assign_modalities(cases)
    _apply_modality_reference(cases)
    _assign_splits(cases)

    ids = [c["case_id"] for c in cases]
    if len(set(ids)) != len(ids):
        dupes = [i for i, n in Counter(ids).items() if n > 1]
        raise ValueError(f"Benchmark generation produced duplicate case_ids: {dupes[:10]}")
    return cases


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_case(case: dict) -> list[str]:
    """Return a list of schema problems for one case (empty list == valid)."""
    problems: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in case:
            problems.append(f"missing field '{field}'")
    if case.get("language") not in LANGUAGES:
        problems.append(f"invalid language {case.get('language')!r}")
    if case.get("modality") not in MODALITIES:
        problems.append(f"invalid modality {case.get('modality')!r}")
    if case.get("expected_label") not in LABELS:
        problems.append(f"invalid expected_label {case.get('expected_label')!r}")
    if case.get("expected_risk_band") not in RISK_BANDS:
        problems.append(f"invalid expected_risk_band {case.get('expected_risk_band')!r}")
    if case.get("difficulty") not in DIFFICULTIES:
        problems.append(f"invalid difficulty {case.get('difficulty')!r}")
    if case.get("source_type") not in SOURCE_TYPES:
        problems.append(f"invalid source_type {case.get('source_type')!r}")
    if case.get("split") not in SPLITS:
        problems.append(f"invalid split {case.get('split')!r}")
    if not isinstance(case.get("expected_signals"), list):
        problems.append("expected_signals must be a list")
    if not str(case.get("text_or_fixture_reference", "")).strip():
        problems.append("empty text_or_fixture_reference")
    # Label / risk-band consistency.
    label, band = case.get("expected_label"), case.get("expected_risk_band")
    if label == "legitimate" and band != "low":
        problems.append("legitimate label must map to risk band 'low'")
    if label == "ambiguous" and band != "medium":
        problems.append("ambiguous label must map to risk band 'medium'")
    if label == "scam" and band not in {"high", "critical"}:
        problems.append("scam label must map to risk band 'high' or 'critical'")
    # Modality-specific fixture requirements.
    if case.get("modality") in MEDIA_KIND:
        fixture = case.get("fixture")
        if not isinstance(fixture, dict) or fixture.get("status") != "requires_real_fixture":
            problems.append(f"{case.get('modality')} case must carry a requires_real_fixture fixture block")
        if not str(case.get("text_or_fixture_reference", "")).startswith("data/evaluation/fixtures/"):
            problems.append("media case text_or_fixture_reference must point at a fixture path")
    else:
        if str(case.get("text_or_fixture_reference", "")).startswith("data/evaluation/fixtures/"):
            problems.append("non-media case must inline its text, not a fixture path")
    # Adversarial integrity fields.
    if case.get("source_type") == "adversarial_pair":
        if not case.get("adversarial_pair_id"):
            problems.append("adversarial case missing adversarial_pair_id")
        if case.get("difficulty") != "hard":
            problems.append("adversarial case difficulty must be 'hard'")
    return problems


# --------------------------------------------------------------------------- #
# Distribution helpers
# --------------------------------------------------------------------------- #
def label_bucket(case: dict) -> str:
    """Mutually exclusive reporting bucket matching TARGET_LABEL_BUCKETS."""
    return "adversarial" if case["source_type"] == "adversarial_pair" else case["expected_label"]


def distributions(cases: list[dict]) -> dict:
    return {
        "total": len(cases),
        "label_bucket": dict(Counter(label_bucket(c) for c in cases)),
        "expected_label": dict(Counter(c["expected_label"] for c in cases)),
        "language": dict(Counter(c["language"] for c in cases)),
        "modality": dict(Counter(c["modality"] for c in cases)),
        "category": dict(Counter(c["category"] for c in cases)),
        "difficulty": dict(Counter(c["difficulty"] for c in cases)),
        "split": dict(Counter(c["split"] for c in cases)),
    }


# --------------------------------------------------------------------------- #
# Quality checks (exact/near duplicates, contradictions, repeated templates)
# --------------------------------------------------------------------------- #
def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def case_evidence_text(case: dict) -> str:
    """Return the authored evidence content of a case.

    For screenshot/audio cases ``text_or_fixture_reference`` is a fixture PATH,
    so the comparable content lives in ``fixture.render_text``. Quality checks
    must compare evidence content, never the (near-identical) fixture paths.
    """
    fixture = case.get("fixture")
    if isinstance(fixture, dict) and fixture.get("render_text"):
        return fixture["render_text"]
    return case.get("text_or_fixture_reference", "")


def _skeleton(text: str) -> str:
    """Strip volatile tokens (numbers, urls, phones) to expose a shared template."""
    text = _normalize(text)
    text = re.sub(r"https?://\S+", "<url>", text)
    text = re.sub(r"\+?\d[\d\s\-]{5,}", "<num>", text)
    text = re.sub(r"\b\d[\d,\.]*\b", "<n>", text)
    return text.strip()


def exact_duplicates(cases: list[dict]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        groups[_normalize(case_evidence_text(case))].append(case["case_id"])
    return {text: ids for text, ids in groups.items() if len(ids) > 1}


def near_duplicate_pairs(cases: list[dict], threshold: float = 0.94) -> list[tuple[str, str, float]]:
    """O(n^2) similarity check within each language to keep it tractable and meaningful."""
    pairs: list[tuple[str, str, float]] = []
    by_language: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        by_language[case["language"]].append(case)
    for members in by_language.values():
        texts = [(c["case_id"], _normalize(case_evidence_text(c))) for c in members]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                (id_a, text_a), (id_b, text_b) = texts[i], texts[j]
                if abs(len(text_a) - len(text_b)) > max(len(text_a), len(text_b)) * 0.4:
                    continue
                ratio = SequenceMatcher(None, text_a, text_b).ratio()
                if ratio >= threshold:
                    pairs.append((id_a, id_b, round(ratio, 4)))
    return pairs


def label_contradictions(cases: list[dict]) -> list[tuple[str, str, str]]:
    """Identical evidence text carrying different expected labels."""
    by_text: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        by_text[_normalize(case_evidence_text(case))].append(case)
    conflicts: list[tuple[str, str, str]] = []
    for text, group in by_text.items():
        labels = {c["expected_label"] for c in group}
        if len(labels) > 1:
            conflicts.append((text, sorted(labels)[0], sorted(labels)[1]))
    return conflicts


def repeated_templates(cases: list[dict], max_repeats: int = 3) -> dict[str, int]:
    """Skeletons reused more than ``max_repeats`` times within a language."""
    counts: Counter = Counter()
    for case in cases:
        counts[(case["language"], _skeleton(case_evidence_text(case)))] += 1
    return {
        f"{lang}::{skeleton}": n
        for (lang, skeleton), n in counts.items()
        if n > max_repeats
    }


def quality_report(cases: list[dict], near_dup_threshold: float = 0.94) -> dict:
    dist = distributions(cases)
    exact = exact_duplicates(cases)
    near = near_duplicate_pairs(cases, near_dup_threshold)
    contradictions = label_contradictions(cases)
    repeats = repeated_templates(cases)
    lang_counts = [dist["language"].get(lang, 0) for lang in LANGUAGES]
    cat_counts = list(dist["category"].values())
    return {
        "total": dist["total"],
        "exact_duplicate_groups": len(exact),
        "exact_duplicate_ids": sorted(i for ids in exact.values() for i in ids),
        "near_duplicate_pair_count": len(near),
        "near_duplicate_pairs": near[:50],
        "label_contradiction_count": len(contradictions),
        "label_contradictions": contradictions,
        "repeated_template_count": len(repeats),
        "repeated_templates": repeats,
        "language_imbalance_spread": max(lang_counts) - min(lang_counts) if lang_counts else 0,
        "category_min": min(cat_counts) if cat_counts else 0,
        "category_max": max(cat_counts) if cat_counts else 0,
        "distinct_category_count": len(cat_counts),
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
import json  # noqa: E402  (kept local to avoid a top-level import cycle in tooling)


def write_jsonl(cases: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(case, ensure_ascii=False, sort_keys=False) for case in cases]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# Pure evaluation metrics (no live model calls; unit-tested)
# --------------------------------------------------------------------------- #
def predicted_class(result: dict) -> str:
    """Map a pipeline/baseline result to scam / legitimate / uncertain / unavailable."""
    assessment = result.get("assessment")
    if not assessment:
        return "unavailable"
    level = assessment.get("risk_level")
    if level in {"high", "critical"}:
        return "scam"
    if level == "low":
        return "legitimate"
    return "uncertain"


def confusion_counts(runs: list[dict]) -> dict:
    """Binary confusion counts over definitive expected+predicted scam/legitimate runs."""
    scored = [
        r for r in runs
        if r.get("expected_label") in {"scam", "legitimate"} and r.get("predicted") in {"scam", "legitimate"}
    ]
    tp = sum(1 for r in scored if r["expected_label"] == "scam" and r["predicted"] == "scam")
    fn = sum(1 for r in scored if r["expected_label"] == "scam" and r["predicted"] == "legitimate")
    fp = sum(1 for r in scored if r["expected_label"] == "legitimate" and r["predicted"] == "scam")
    tn = sum(1 for r in scored if r["expected_label"] == "legitimate" and r["predicted"] == "legitimate")
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "scored": len(scored)}


def _safe_div(a: float, b: float) -> float | None:
    return (a / b) if b else None


def classification_metrics(counts: dict) -> dict:
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
    scored = counts["scored"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if precision and recall else (0.0 if precision is not None and recall is not None else None)
    return {
        "accuracy": _safe_div(tp + tn, scored),
        "scam_recall": recall,
        "legitimate_specificity": _safe_div(tn, tn + fp),
        "false_positive_rate": _safe_div(fp, fp + tn),
        "precision": precision,
        "f1": f1,
    }


def _breakdown(runs: list[dict], field: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        groups[str(run.get(field, "unknown"))].append(run)
    out: dict[str, dict] = {}
    for key in sorted(groups):
        members = groups[key]
        counts = confusion_counts(members)
        metrics = classification_metrics(counts)
        fixture_unavailable = sum(1 for r in members if r.get("status") == "fixture_unavailable")
        # Model failure only: an unavailable prediction that is NOT a missing fixture.
        model_unavailable = sum(
            1 for r in members
            if r.get("predicted") == "unavailable" and r.get("status") != "fixture_unavailable"
        )
        out[key] = {
            "total": len(members),
            "scored": counts["scored"],
            **counts,
            **metrics,
            "uncertain": sum(1 for r in members if r.get("predicted") == "uncertain"),
            "fixture_unavailable": fixture_unavailable,
            "unavailable": model_unavailable,
            "executed": len(members) - fixture_unavailable,
        }
    return out


def summarize_runs(runs: list[dict]) -> dict:
    """Full metric summary with breakdowns, matching the evaluator's report.

    Three non-definitive outcomes are kept strictly separate and never conflated:
    - ``fixture_unavailable``: a screenshot/audio case whose real fixture file is
      absent. It was NOT executed and says nothing about the model.
    - ``predicted_unavailable``: the model ran but produced no assessment
      (error/timeout). This is a model failure, not model uncertainty.
    - ``predicted_uncertain``: the model produced a ``medium`` band, i.e. a real
      calibrated abstention. Only these count toward ``abstention_rate``.
    """
    total = len(runs)
    counts = confusion_counts(runs)
    metrics = classification_metrics(counts)
    definitive = counts["scored"]
    ambiguous_expected = sum(1 for r in runs if r.get("expected_label") == "ambiguous")

    fixture_unavailable = sum(1 for r in runs if r.get("status") == "fixture_unavailable")
    model_unavailable = sum(
        1 for r in runs
        if r.get("predicted") == "unavailable" and r.get("status") != "fixture_unavailable"
    )
    predicted_uncertain = sum(1 for r in runs if r.get("predicted") == "uncertain")

    # Cases that actually entered the model, and cases where it returned a risk band.
    executed = total - fixture_unavailable
    model_responded = executed - model_unavailable

    return {
        "counts": counts,
        "metrics": metrics,
        "total": total,
        "definitive": definitive,
        "ambiguous_expected": ambiguous_expected,
        "fixture_unavailable": fixture_unavailable,
        "executed": executed,
        "execution_coverage": _safe_div(executed, total),
        "model_responded": model_responded,
        # ``predicted_unavailable`` now means MODEL failure only (fixtures excluded).
        "predicted_unavailable": model_unavailable,
        "predicted_uncertain": predicted_uncertain,
        # Abstention counts ONLY real model uncertainty, over cases the model answered.
        "abstention_rate": _safe_div(predicted_uncertain, model_responded),
        "breakdowns": {
            "language": _breakdown(runs, "language"),
            "category": _breakdown(runs, "category"),
            "modality": _breakdown(runs, "modality"),
            "difficulty": _breakdown(runs, "difficulty"),
        },
    }
