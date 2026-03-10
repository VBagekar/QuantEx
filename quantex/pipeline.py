"""
pipeline.py

The QuantEx pipeline — single entry point that orchestrates all modules.

Usage:
    from quantex.pipeline import run_pipeline

    facts = run_pipeline("Apple reported revenue of $117 billion in Q1 2024.")
    for fact in facts:
        print(fact)

Output:
    QuantExResult(
        entity       = "Apple",
        entity_label = "ORG",
        value        = 117.0,
        unit         = "billion dollar",
        unit_type    = "currency",
        context      = "revenue",
        raw_quantity = "$117 billion",
        sentence     = "Apple reported revenue of $117 billion in Q1 2024.",
        link_method  = "dep_tree"
    )
"""

from dataclasses import dataclass
from typing import Optional
from quantex.linker import link_quantities_to_entities
from quantex.context_extractor import extract_context


# ─────────────────────────────────────────────
# FINAL OUTPUT DATA CLASS
# ─────────────────────────────────────────────

@dataclass
class QuantExResult:
    entity: str
    entity_label: str
    value: float
    unit: Optional[str]
    unit_type: Optional[str]
    context: Optional[str]       # ← NEW: "revenue", "population", "scored"
    raw_quantity: str
    sentence: str
    link_method: str

    def __str__(self):
        return (
            f"[{self.entity_label}] {self.entity} "
            f"→ {self.value} {self.unit or ''} "
            f"| context: '{self.context or 'N/A'}' "
            f"| via {self.link_method}"
        )


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(sentence: str) -> list[QuantExResult]:
    """
    Full QuantEx pipeline for a single sentence.

    Steps:
        1. link_quantities_to_entities() — Commit 2+3+4
        2. extract_context()             — Commit 5
        3. Package into QuantExResult
    """
    linked_facts = link_quantities_to_entities(sentence)
    results = []

    for fact in linked_facts:
        context = extract_context(
            sentence=sentence,
            entity_text=fact.entity,
            quantity_raw=fact.raw_quantity,
            unit=fact.quantity_unit, 
        )
        results.append(QuantExResult(
            entity=fact.entity,
            entity_label=fact.entity_label,
            value=fact.quantity_value,
            unit=fact.quantity_unit,
            unit_type=fact.quantity_unit_type,
            context=context,
            raw_quantity=fact.raw_quantity,
            sentence=sentence,
            link_method=fact.link_method,
        ))

    return results