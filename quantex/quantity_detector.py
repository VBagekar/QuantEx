"""
quantity_detector.py

Two-phase quantity detection:

Phase 1 — Regex: finds numeric values and KNOWN units (meter, kg, percent, $)
          from units.json (kept only for normalized forms + currency symbols)

Phase 2 — spaCy nummod: for ANY number, find the noun it modifies via the
          dependency tree. This auto-discovers domain-specific units like
          "runs", "neurons", "vehicles" WITHOUT manual addition.

          "183 runs" → runs (nummod → 183) → unit = "run"
          "86 billion neurons" → neurons (nummod → billion → 86) → unit = "neuron"

This means units.json is now only needed for:
  - Normalization: "metres" → "meter"
  - Currency symbols: $ → dollar
  - Scale words: million, billion (these modify nouns, not standalone units)
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import spacy


# ─────────────────────────────────────────────
# 1. DATA CLASS
# ─────────────────────────────────────────────

@dataclass
class QuantitySpan:
    value: float
    unit: Optional[str]
    unit_type: Optional[str]
    raw_text: str
    char_start: int
    char_end: int


# ─────────────────────────────────────────────
# 2. LOAD UNIT DATABASE (now only for normalization)
# ─────────────────────────────────────────────

def _load_units():
    json_path = Path(__file__).parent.parent / "data" / "units.json"
    with open(json_path, "r") as f:
        data = json.load(f)
    return {e["pattern"].lower(): (e["norm"], e["type"]) for e in data["units"]}

UNIT_LOOKUP = _load_units()
CURRENCY_SYMBOLS = {"$": "dollar", "₹": "rupee", "€": "euro", "£": "pound"}
SCALE_WORDS = {"million", "billion", "trillion", "thousand", "lakh", "crore"}

# ─────────────────────────────────────────────
# 3. LOAD spaCy (reuse from entity_extractor if loaded)
# ─────────────────────────────────────────────

def _load_model():
    try:
        return spacy.load("en_core_web_trf")
    except OSError:
        return spacy.load("en_core_web_lg")

nlp = _load_model()


# ─────────────────────────────────────────────
# 4. REGEX — finds numbers + currency symbols only
# ─────────────────────────────────────────────

NUMBER_PATTERN = re.compile(
    r"""
    (?P<currency>[$₹€£])?
    (?P<number>
        \d{1,3}(?:,\d{3})+     # comma-formatted: 1,400,000
        |\d+(?:\.\d+)?          # plain integer or decimal
    )
    """,
    re.VERBOSE
)


# ─────────────────────────────────────────────
# 5. FIND UNIT VIA nummod DEPENDENCY
#
# In a dependency parse, numbers modify nouns via "nummod".
# We walk UP from the number token to find what noun it modifies.
#
# "183 runs":       runs <--nummod-- 183   → unit = runs
# "1.4 billion people":
#    people <--nummod-- billion <--nummod-- 1.4  → unit = people
# "828 meters":     meters <--nummod-- 828  → unit = meters
# "$117 billion":   no nummod noun → unit from currency symbol
# ─────────────────────────────────────────────

def _find_unit_from_dep(doc, number_token) -> Optional[tuple[str, str]]:
    """
    Walk up the dependency tree from number_token to find its head noun.
    Preserves scale words (billion/million) as prefix to the unit.
    """
    token = number_token
    scale_prefix = None

    for _ in range(2):
        head = token.head
        if head == token:
            break

        # If head is a scale word, remember it and keep walking
        if head.lemma_.lower() in SCALE_WORDS:
            scale_prefix = head.lemma_.lower()
            token = head
            continue

        # Found the unit noun
        if head.pos_ in {"NOUN", "PROPN", "VERB"}:
            lemma = head.lemma_.lower()
            if lemma in UNIT_LOOKUP:
                norm, utype = UNIT_LOOKUP[lemma]
            else:
                norm = lemma
                utype = _infer_type(lemma)

            # Prepend scale if we passed through one
            if scale_prefix:
                norm = f"{scale_prefix} {norm}"

            return (norm, utype)

        token = head

    # If we only found a scale word with no noun head, return scale alone
    if scale_prefix:
        return UNIT_LOOKUP.get(scale_prefix, (scale_prefix, "scale"))

    return None

def _infer_type(lemma: str) -> str:
    """
    Infer unit type from lemma using simple heuristics.
    This runs only for auto-discovered units not in units.json.
    """
    # You can extend this dict freely — it's just for labeling
    type_hints = {
        "run": "sports", "goal": "sports", "wicket": "sports",
        "neuron": "biology", "cell": "biology",
        "vehicle": "count", "user": "count", "employee": "count",
        "person": "count", "people": "count",
    }
    return type_hints.get(lemma, "count")  # default to "count" for unknowns


# ─────────────────────────────────────────────
# 6. MAIN FUNCTION
# ─────────────────────────────────────────────

def detect_quantities(text: str) -> list[QuantitySpan]:
    """
    Detect quantities using regex for values + spaCy nummod for units.

    Example:
        "Kohli scored 183 runs" → QuantitySpan(183, "run", "sports", ...)
        "$117 billion revenue"  → QuantitySpan(117, "billion dollar", "currency", ...)
        "8849 meters tall"      → QuantitySpan(8849, "meter", "length", ...)
        "86 billion neurons"    → QuantitySpan(86, "billion neuron", "biology", ...)
    """
    doc = nlp(text)
    results = []

    for match in NUMBER_PATTERN.finditer(text):
        raw_number = match.group("number")
        currency_sym = match.group("currency")
        if not raw_number:
            continue

        value = float(raw_number.replace(",", ""))

        # Skip standalone years
        if 1800 <= value <= 2100 and not currency_sym:
            continue

        # Find the corresponding spaCy token
        num_token = None
        for token in doc:
            if token.idx == match.start("number"):
                num_token = token
                break

        if num_token is None:
            continue

        # Resolve unit
        if currency_sym:
            base_unit = CURRENCY_SYMBOLS[currency_sym]
            # Check if next word is a scale (e.g., "$117 billion")
            scale = None
            for token in doc:
                if token.idx == match.end() + 1 and token.lemma_.lower() in SCALE_WORDS:
                    scale = token.lemma_.lower()
                    break
            unit = f"{scale} {base_unit}" if scale else base_unit
            unit_type = "currency"
        else:
            # Use nummod dependency to find unit
            dep_result = _find_unit_from_dep(doc, num_token)
            if dep_result:
                unit, unit_type = dep_result
            else:
                # Last resort: check if next token is a known unit
                next_tokens = [t for t in doc if t.idx > num_token.idx]
                if next_tokens and next_tokens[0].lemma_.lower() in UNIT_LOOKUP:
                    unit, unit_type = UNIT_LOOKUP[next_tokens[0].lemma_.lower()]
                else:
                    continue  # skip bare numbers

        # Build clean raw_text
        # For currency, raw_text should be symbol + number + scale only (not repeat "dollar")
        if currency_sym:
            scale_word = unit.split()[0] if unit and unit.split()[0] in SCALE_WORDS else ""
            raw_text = f"{currency_sym}{raw_number} {scale_word}".strip()
        else:
            raw_text = f"{raw_number} {unit}".strip()

        results.append(QuantitySpan(
            value=value,
            unit=unit,
            unit_type=unit_type,
            raw_text=raw_text,
            char_start=match.start(),
            char_end=match.end(),
        ))

    return results