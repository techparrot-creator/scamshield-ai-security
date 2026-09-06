---
title: ScamShield AI
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: gradio
python_version: "3.11"
app_file: app.py
fullWidth: true
header: mini
short_description: Bilingual multimodal scam investigation with LangGraph and advanced verified RAG.
models:
  - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
tags:
  - langgraph
  - gradio
  - rag
  - chromadb
  - cybersecurity
  - urdu
  - scam-detection
preload_from_hub:
  - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
---

# ScamShield AI — Hackathon Build v1.0

> **Qoder IDE-first repository:** open this project directly in Qoder IDE. Qoder automatically recognizes `AGENTS.md`, project-specific guidance is included under `.qoder/rules/`, and a complete setup guide is in `QODER_START_HERE.md`.

## Qoder IDE quick start

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\qoder_setup.ps1
```

Then add `GOOGLE_API_KEY` to `.env`, run `python scripts\smoke_test.py`, and start the product with `python app.py`. Use a second Qoder terminal with `langgraph dev` for the Studio graph.


ScamShield AI is a bilingual Urdu, Roman Urdu, and English cyber-safety agent for suspicious social-media messages, links, screenshots, emails, and voice notes. It uses LangGraph to orchestrate evidence processing, URL inspection, advanced retrieval, conditional query expansion, structured risk assessment, recovery guidance, and report generation.

## What makes this build technically strong

- **LangGraph:** visible conditional workflow rather than a single prompt.
- **Gemini:** structured reasoning plus image/audio evidence extraction.
- **Hugging Face:** multilingual semantic embeddings.
- **ChromaDB:** vector search over verified cyber-safety knowledge.
- **BM25:** exact keyword retrieval for indicators such as OTP, CNIC, WhatsApp, JazzCash, crypto, and job fees.
- **MMR + RRF:** diverse retrieval and semantic/keyword result fusion.
- **Metadata-aware retrieval:** Pakistan and scam-topic boosts.
- **Conditional Multi-Query:** only activates when first-pass retrieval confidence is weak.
- **Parent Context:** precise child-section search with broader verified context restored before generation.
- **Gradio:** multimodal product UI ready for Hugging Face Spaces.
- **Privacy:** exported reports redact likely OTPs, CNICs, card-like numbers, phone numbers, and emails.

## Project structure

```text
scamshield-ai-hackathon/
├── app.py
├── langgraph.json
├── requirements.txt
├── pyproject.toml
├── .env.example
├── knowledge/
│   └── sources.json
├── data/
│   └── eval_cases.jsonl
├── scamshield/
│   ├── config.py
│   ├── graph.py
│   ├── media.py
│   ├── privacy.py
│   ├── prompts.py
│   ├── query_planner.py
│   ├── retrieval.py
│   ├── schemas.py
│   └── url_tools.py
├── scripts/
│   ├── build_index.py
│   ├── doctor.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_system.py
│   ├── multimodal_smoke_test.py
│   └── smoke_test.py
├── tests/
└── docs/
```

# Local setup — Windows / VS Code

## 1. Open this folder in VS Code

## 2. Create the environment

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Or run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
```

## 3. Configure secrets

```powershell
Copy-Item .env.example .env
code .env
```

Set at minimum:

```env
GOOGLE_API_KEY=your_real_google_ai_studio_key
```

Never commit `.env`.

## 4. Warm and inspect the RAG index

```powershell
python scripts\build_index.py
```

The first run downloads the multilingual Hugging Face model and can take longer than later runs.

## 5. Run tests

```powershell
pytest -q
```

## 6. Run the product UI

```powershell
python app.py
```

Open the Gradio URL, normally `http://127.0.0.1:7860`.

## 7. Run LangGraph Studio

In another activated terminal:

```powershell
langgraph dev
```

The graph name is `scamshield`.

## 8. Evaluate retrieval

```powershell
python scripts\evaluate_retrieval.py
```

This checks whether known scam examples retrieve at least one expected official source.

# Technical Evaluation

A judge-facing summary of how to reproduce every technical claim in this build.

## Unit tests (offline, no API key needed)

```powershell
python -m pytest -q
```

Covers OTP-direction calibration, privacy redaction, URL heuristics, query planning,
the resilient Gemini retry helper, pipeline-trace privacy/formatting, graceful model
failure rendering, and the mocked multimodal pipeline path.

## Retrieval evaluation (offline; hit@k over the verified RAG knowledge base)

```powershell
python scripts\evaluate_retrieval.py
```

## System classification evaluation (LIVE GEMINI CALLS, requires GOOGLE_API_KEY)

```powershell
python scripts\evaluate_system.py
```

Runs the full LangGraph + RAG + Gemini pipeline over `data/system_eval_cases.jsonl`,
a curated/synthetic dataset kept separate from the RAG knowledge base. It includes
balanced scam / legitimate / insufficient-evidence cases in English, Urdu, and Roman
Urdu, plus adversarial pairs (service-sends-code vs code-requested, real Google notice
vs fake Google support, normal job vs advance-fee job, transaction alert vs PIN/OTP
request, delivery notice vs fake delivery fee). Reported metrics: total cases,
accuracy, scam recall, legitimate specificity, false-positive rate, confusion counts,
and a per-language breakdown. Label mapping is documented in the script header:
`high/critical -> scam`, `low -> legitimate`, `medium -> uncertain` (excluded from
binary metrics and reported separately), missing assessment -> `unavailable`
(excluded). No metric is reported for cases the evaluator cannot reliably map.

## Multimodal proof (LIVE GEMINI CALLS, requires GOOGLE_API_KEY)

```powershell
python scripts\multimodal_smoke_test.py --image path\to\sample.png
python scripts\multimodal_smoke_test.py --audio path\to\sample.wav
python scripts\multimodal_smoke_test.py --image shot.png --audio note.wav
```

For each supplied file it prints: detected modality, extraction status and extracted
character count, retrieval executed yes/no, retrieval confidence, Multi-Query
activation, retrieved source IDs, assessment availability, and PASS/FAIL. Only safe
technical results are printed — raw media contents are never shown. Unsupported or
corrupt files fail gracefully with a FAIL row instead of a traceback.

## Safe pipeline trace

Every analysis produces a structured `pipeline_trace` in LangGraph state, shown in the
Gradio **Technical Analysis Trace** tab. It records only observable system-level
events: input type, evidence-extraction status and extracted length, detected
indicator labels (e.g. `otp_request`, `urgency`, `payment_request`), URL count and
structural warning labels, semantic/BM25 execution status, fused result count,
retrieval confidence vs threshold, conditional Multi-Query activation, retrieved
source IDs/titles, parent-context restoration status, Gemini status, final risk
level/category, and elapsed milliseconds per stage. It never contains raw OTPs,
CNICs, passwords, card numbers, phone numbers, full emails, full user messages, or
hidden model reasoning / chain-of-thought.

## Graceful model failure behavior

All Gemini calls go through a single resilient helper (`scamshield/llm_client.py`):
transient failures (network disconnects, connection errors, read timeouts, 429/5xx)
are retried up to 3 total attempts with exponential backoff (~1s, 2s); authentication
and configuration errors (invalid key, invalid request, unsupported model, permission
denied) are NOT blindly retried; secrets are redacted from every logged error. If the
model is still unavailable, the pipeline does not crash: the UI shows
"AI assessment is temporarily unavailable. Extracted evidence and verified safety
sources are shown below. Please retry shortly." while preserving evidence extraction,
URL findings, and retrieved RAG sources. No risk classification is fabricated when the
model never produced one.

# Evaluation benchmark (hackathon-grade, multilingual)

A structured, reproducible benchmark for measuring ScamShield's scam / legitimate /
uncertain decisions across languages, modalities, and scam categories. It is an
**evaluation harness only** and is deliberately kept **separate from the RAG
knowledge base** (`knowledge/sources.json`) so no benchmark case is ever indexed
for retrieval (no evaluation leakage).

## Purpose

Give judges a transparent, deterministic way to reproduce classification quality,
inspect calibration and abstention, compare against a simple baseline, and audit
dataset quality — without claiming production-grade or universal accuracy.

## Dataset composition (600 cases)

All content is **synthetic / curated** and privacy-safe. There is **no real victim
data**. Slot values (amounts, codes, phone numbers, URLs, bank names) are synthetic
and drawn from fixed pools so builds are fully deterministic.

- **Labels (mutually exclusive buckets):** 240 scam, 210 legitimate, 90 ambiguous,
  60 adversarial = 600. (Adversarial members still carry a real `scam`/`legitimate`
  label for scoring; folded back, expected labels are 270 scam / 240 legitimate / 90 ambiguous.)
- **Languages:** 200 English, 200 Roman Urdu, 200 Urdu.
- **Modalities:** 360 text, 90 email/link, 90 screenshot, 60 audio.
- **Scam categories (15):** OTP/PIN theft, phishing/smishing, bank impersonation,
  WhatsApp takeover, fake job/task, fake seller/buyer, investment/crypto/forex,
  courier impersonation, government/police/support impersonation, remote access,
  lottery/prize/advance-fee, romance, blackmail/extortion, payment recovery,
  account recovery.
- **Legitimate negative controls (10):** do-not-share OTP notices, transaction alerts,
  password resets, courier updates, ordinary job offers, marketplace chats, security
  notifications, support messages, personal chats, payment receipts.
- **Adversarial pairs (30):** scam/legitimate pairs that differ in **critical intent**
  (who must reveal a secret, who requests money, which channel), not a single swapped word.

Each case carries: `case_id`, `language`, `modality`, `category`, `expected_label`,
`expected_risk_band`, `text_or_fixture_reference`, `expected_signals`, `difficulty`,
`source_type`, `notes`, `split`, and (for media) a `fixture` block and
`adversarial_pair_id` (for pairs).

## Files

```text
data/evaluation/scamshield_benchmark.jsonl          # all 600 cases
data/evaluation/scamshield_benchmark_dev.jsonl      # development / calibration split (~402)
data/evaluation/scamshield_benchmark_holdout.jsonl  # LOCKED holdout split (~198)
data/evaluation/results/                            # exported JSON/CSV reports
```

## Development vs locked holdout

The split is **deterministic and stratified** (by label bucket and language). The
`development` split is for calibration and inspection. The `holdout` split is
**locked**: it must not be used for threshold tuning, and adversarial pairs always
stay together in one split. Use holdout only for the final report and the ablation.

## Build, evaluate, and audit

```powershell
# 1) Build the deterministic benchmark + split files (offline, no API key)
python scripts\build_benchmark.py

# 2) Dataset-quality report: duplicates, contradictions, imbalance (offline)
python scripts\benchmark_quality_report.py

# 3) Evaluate the FULL pipeline (LIVE Gemini, requires GOOGLE_API_KEY)
python scripts\evaluate_benchmark.py                 # development split
python scripts\evaluate_benchmark.py --split holdout # locked holdout

# 3b) Balanced, reproducible holdout SAMPLE (deterministic stratified sampling).
#     --stratified balances language x expected-label x modality; --seed makes it
#     reproducible; --executable-only drops screenshot/audio cases whose real
#     fixture file is missing (they would otherwise be fixture_unavailable).
python scripts\evaluate_benchmark.py --split holdout --stratified --executable-only --limit 30 --seed 20260904

# 4) Baseline vs full ablation on the same locked holdout (LIVE Gemini)
python scripts\evaluate_baseline_ablation.py
```

## Metrics

The evaluator reports: total, executed, execution coverage, model-responded,
definitive (scored), ambiguous-expected, **TP/TN/FP/FN**, accuracy, scam recall,
legitimate specificity, false-positive rate, precision, F1, and the
uncertainty/abstention rate — plus a **confusion matrix** and breakdowns by
**language, category, modality, and difficulty**. Results export to JSON and CSV.

Three non-definitive outcomes are kept strictly separate and never conflated:
- **fixture-unavailable** — a screenshot/audio case whose real fixture file is
  absent; it was NOT executed and says nothing about the model.
- **predicted unavailable** — the model ran but returned no assessment
  (error/timeout); this is a model failure, not model uncertainty.
- **predicted uncertain** — a real `medium`-band abstention. Only these count
  toward the abstention rate, computed as `uncertain / model-responded`.

Documented label mapping (identical to `evaluate_system.py`, no fabricated metrics):
`high/critical -> scam`, `low -> legitimate`, `medium -> uncertain` (calibrated
abstention, excluded from binary metrics), missing assessment -> `unavailable`
(excluded). Ambiguous-expected cases have no binary ground truth and are reported as
a calibration check, never counted in accuracy.

## Baseline / ablation

`evaluate_baseline_ablation.py` runs, on the **same** holdout cases:
- **A. Gemini-only baseline:** raw evidence -> one structured assessment (no RAG, no
  URL heuristics, no query planning, no LangGraph).
- **B. Full ScamShield:** the complete LangGraph + advanced hybrid RAG pipeline.

Both are scored with the identical mapping. If API quota/outage prevents completion,
the script reports **INCOMPLETE EXECUTION** with exact executed counts and exits
non-zero — it never fabricates results.

## Multimodal (screenshot / audio) honesty

The project has **no offline synthetic image/audio generation path** (media extraction
requires live Gemini over real bytes). Screenshot/audio cases therefore ship with a
`fixture` block (`status: requires_real_fixture`, the intended `render_text`, and the
target `path`). Until a real fixture file is placed at that path, these cases are
reported as `fixture_unavailable` and **excluded** — a multimodal **PASS is never
claimed unless the actual path executes**. Provide fixtures by saving a real
screenshot/voice note containing `render_text` at the recorded `path`.

## Limitations

- Synthetic, curated content authored for safety evaluation; it is **not** a
  representative sample of real-world traffic.
- **No claim of population-scale, production-level, or universal scam-detection accuracy.**
- Metrics depend on live Gemini behavior, which can change with model/version and quota.
- Screenshot/audio scores require user-supplied real fixtures; without them those
  modalities are reported as not executed rather than passing.
- Ambiguous cases intentionally have no binary ground truth; they measure calibration,
  not accuracy.

# Hugging Face Spaces deployment

1. Create a **Gradio** Space.
2. Upload this repo except `.env`, `.venv`, caches, and local reports.
3. Add `GOOGLE_API_KEY` under **Settings → Secrets**.
4. Keep model/retrieval configuration as public Variables if desired.
5. Wait for build and embedding-model preload.

# Qoder

Qoder is not required by the app. Use it as an engineering assistant while keeping the repo portable. See `docs/qoder-guide.md`.

# Demo inputs

### Roman Urdu OTP impersonation

```text
Main bank security team se hun. Apna OTP foran bhejo warna account block ho jayega.
```

### Fake job

```text
We found your CV. Earn PKR 20,000 daily from home. Pay a training fee today to activate the job.
```

### Recovery case

```text
I sent money to a seller from a wallet and now they blocked me. What should I do?
```

# Safety boundary

ScamShield provides decision support, not legal determination. It must not label a person as definitely criminal, guarantee that a URL is safe, or guarantee money recovery. External reporting/submission should remain user-controlled.
