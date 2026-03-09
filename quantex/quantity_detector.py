"""
quantity_detector.py

Detects and normalizes quantity expressions from raw text.

A "quantity" is any numeric expression with an optional unit.
Examples: "1.4 billion", "$3.7 trillion", "828 meters", "85 percent"

Each detected quantity is returned as a QuantitySpan dataclass containing:
  - value       : the numeric value as a float
  - unit        : normalized unit string (e.g., "meter", "kilogram")
  - unit_type   : category of unit (e.g., "length", "currency", "count")
  - raw_text    : the original matched string from the sentence
  - char_start  : character index where the match starts (used later for linking)
  - char_end    : character index where the match ends
"""

import re
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
# 1. DATA CLASS — What a detected quantity looks like
# ─────────────────────────────────────────────

@dataclass
class QuantitySpan:
    value: float
    unit: Optional[str]       # e.g., "meter", "billion", "percent"
    unit_type: Optional[str]  # e.g., "length", "count", "currency"
    raw_text: str             # e.g., "1.4 billion"
    char_start: int           # position in original sentence
    char_end: int


# ─────────────────────────────────────────────
# 2. UNIT DICTIONARY — Maps raw words → (normalized_unit, unit_type)
#
# Why normalize? "metres", "meter", "mtr" should all become "meter"
# so downstream code doesn't need to handle every spelling variant.
# ─────────────────────────────────────────────

UNIT_MAP = {
    # Length
    "meter": ("meter", "length"),
    "meters": ("meter", "length"),
    "metre": ("meter", "length"),
    "metres": ("meter", "length"),
    "km": ("kilometer", "length"),
    "kms": ("kilometer", "length"),
    "kilometer": ("kilometer", "length"),
    "kilometers": ("kilometer", "length"),
    "kilometre": ("kilometer", "length"),
    "kilometres": ("kilometer", "length"),
    "mile": ("mile", "length"),
    "miles": ("mile", "length"),
    "cm": ("centimeter", "length"),
    "centimeter": ("centimeter", "length"),
    "centimeters": ("centimeter", "length"),
    "ft": ("foot", "length"),
    "feet": ("foot", "length"),
    "foot": ("foot", "length"),

    # Weight / Mass
    "kg": ("kilogram", "mass"),
    "kgs": ("kilogram", "mass"),
    "kilogram": ("kilogram", "mass"),
    "kilograms": ("kilogram", "mass"),
    "tonne": ("tonne", "mass"),
    "tonnes": ("tonne", "mass"),
    "ton": ("tonne", "mass"),
    "tons": ("tonne", "mass"),
    "gram": ("gram", "mass"),
    "grams": ("gram", "mass"),
    "pound": ("pound", "mass"),
    "pounds": ("pound", "mass"),

    # Scale / Count (very common in factual sentences)
    "million": ("million", "scale"),
    "billion": ("billion", "scale"),
    "trillion": ("trillion", "scale"),
    "thousand": ("thousand", "scale"),
    "lakh": ("lakh", "scale"),
    "crore": ("crore", "scale"),

    # Currency
    "dollar": ("dollar", "currency"),
    "dollars": ("dollar", "currency"),
    "rupee": ("rupee", "currency"),
    "rupees": ("rupee", "currency"),
    "euro": ("euro", "currency"),
    "euros": ("euro", "currency"),

    # Percentage
    "percent": ("percent", "ratio"),
    "%": ("percent", "ratio"),

    # Time
    "year": ("year", "time"),
    "years": ("year", "time"),
    "month": ("month", "time"),
    "months": ("month", "time"),
    "day": ("day", "time"),
    "days": ("day", "time"),
    "hour": ("hour", "time"),
    "hours": ("hour", "time"),

    # Data
    "gb": ("gigabyte", "data"),
    "tb": ("terabyte", "data"),
    "mb": ("megabyte", "data"),
}

# Currency symbols that appear BEFORE the number (e.g., $3.7 trillion, ₹500)
CURRENCY_SYMBOLS = {
    "$": "dollar",
    "₹": "rupee",
    "€": "euro",
    "£": "pound",
}


# ─────────────────────────────────────────────
# 3. REGEX PATTERNS
#
# This is the core of detection. We use regex to scan text for
# numeric expressions. Let's break down the main pattern:
#
#  (?:[$₹€£])?         → optional currency symbol before number
#  \d{1,3}(?:,\d{3})* → handles comma-formatted numbers like 1,400,000
#  (?:\.\d+)?          → optional decimal part
#  (?:\s+UNIT_WORDS)?  → optional unit word after number (with optional scale)
# ─────────────────────────────────────────────

# All unit words joined into a regex alternation
_unit_words = "|".join(
    sorted(UNIT_MAP.keys(), key=len, reverse=True)  # longest first to avoid partial matches
)

# The master pattern:
# Captures: (currency_symbol?, number, scale_word?, unit_word?)
QUANTITY_PATTERN = re.compile(
    r"""
    (?P<currency>[$₹€£])?               # optional currency symbol
    (?P<number>
        \d{1,3}(?:,\d{3})*             # integer, possibly comma-formatted
        (?:\.\d+)?                     # optional decimal
    )
    (?:                                 # optional unit block
        \s*
        (?P<unit>""" + _unit_words + r""")
        (?:                             # optional secondary scale (e.g., "billion dollars")
            \s+
            (?P<unit2>""" + _unit_words + r""")
        )?
    )?
    """,
    re.VERBOSE | re.IGNORECASE,
)


# ─────────────────────────────────────────────
# 4. HELPER — Parse a raw number string like "1,400,000" → 1400000.0
# ─────────────────────────────────────────────

def _parse_number(raw: str) -> float:
    return float(raw.replace(",", ""))


# ─────────────────────────────────────────────
# 5. MAIN FUNCTION — detect_quantities(text) → list of QuantitySpan
# ─────────────────────────────────────────────

def detect_quantities(text: str) -> list[QuantitySpan]:
    """
    Scan a sentence and return all detected QuantitySpan objects.

    Example:
        detect_quantities("India has 1.4 billion people and GDP of $3.7 trillion")
        → [QuantitySpan(1.4, 'billion', 'scale', '1.4 billion', ...),
           QuantitySpan(3.7, 'trillion dollar', 'currency', '$3.7 trillion', ...)]
    """
    results = []

    for match in QUANTITY_PATTERN.finditer(text):
        raw_number = match.group("number")
        if not raw_number:
            continue

        value = _parse_number(raw_number)

        # Resolve unit: prefer unit2 if it's a currency/mass (more specific)
        # e.g., "3.7 trillion dollars" → unit="trillion", unit2="dollars"
        raw_unit = match.group("unit")
        raw_unit2 = match.group("unit2")
        currency_sym = match.group("currency")

        # Determine final unit and type
        if currency_sym:
            # "$3.7" → unit comes from symbol
            base_unit = CURRENCY_SYMBOLS[currency_sym]
            # If there's also a scale word like "trillion", combine them
            if raw_unit and UNIT_MAP.get(raw_unit.lower(), ("", ""))[1] == "scale":
                unit = f"{UNIT_MAP[raw_unit.lower()][0]} {base_unit}"
                unit_type = "currency"
            else:
                unit = base_unit
                unit_type = "currency"
        elif raw_unit:
            unit_info = UNIT_MAP.get(raw_unit.lower())
            if unit_info:
                unit, unit_type = unit_info
                # If there's a second unit (e.g., "billion neurons" → scale+count)
                if raw_unit2:
                    unit2_info = UNIT_MAP.get(raw_unit2.lower())
                    if unit2_info:
                        unit = f"{unit} {unit2_info[0]}"
            else:
                unit, unit_type = raw_unit, "unknown"
        else:
            unit, unit_type = None, None

        # Only include matches that have a unit OR a meaningful number
        # (skip bare numbers like years "2023" that have no unit context)
        if unit is None and value < 1000:
            continue

        raw_text = match.group(0).strip()
        # Include the currency symbol in raw_text if present
        start = match.start()
        if currency_sym:
            raw_text = currency_sym + raw_text.lstrip()

        results.append(QuantitySpan(
            value=value,
            unit=unit,
            unit_type=unit_type,
            raw_text=raw_text,
            char_start=start,
            char_end=match.end(),
        ))

    return results