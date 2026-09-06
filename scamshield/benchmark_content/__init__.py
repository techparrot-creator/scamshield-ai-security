"""Benchmark content package.

Curated/synthetic message templates for the ScamShield evaluation benchmark.
These are authored fixtures, NOT RAG knowledge and NOT real victim data. Each
module exposes category dictionaries consumed by :mod:`scamshield.benchmark`.

All content is synthetic and written for safety evaluation only. Slot tokens
such as ``{amount}``, ``{name}``, ``{platform}``, ``{deadline}``, ``{code}``,
``{bank}``, ``{url}`` and ``{phone}`` are filled from fixed pools at build time
so that generated cases are deterministic and privacy-safe.
"""
from __future__ import annotations

from scamshield.benchmark_content.adversarial import ADVERSARIAL_PAIRS
from scamshield.benchmark_content.ambiguous import AMBIGUOUS_CATEGORIES
from scamshield.benchmark_content.legitimate import LEGITIMATE_CATEGORIES
from scamshield.benchmark_content.scam import SCAM_CATEGORIES

__all__ = [
    "SCAM_CATEGORIES",
    "LEGITIMATE_CATEGORIES",
    "AMBIGUOUS_CATEGORIES",
    "ADVERSARIAL_PAIRS",
]
