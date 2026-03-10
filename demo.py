"""
demo.py

QuantEx — Live demonstration script.

Runs the full pipeline on a diverse set of sentences and exports
results to data/output.csv for inspection.

Usage:
    python demo.py
"""

import csv
import os
from quantex.pipeline import run_pipeline

# ─────────────────────────────────────────────
# DEMO SENTENCES — diverse domains
# ─────────────────────────────────────────────

SENTENCES = [
    # Sports
    "Virat Kohli scored 183 runs against Pakistan in a test match.",
    "Rohit Sharma hit 264 runs in a single ODI innings.",
    "Neeraj Chopra threw the javelin 87.58 meters to win gold.",

    # Geography
    "Mount Everest is 8849 meters above sea level.",
    "The Amazon River stretches approximately 6400 kilometers.",
    "The Sahara Desert covers about 9.2 million square kilometers.",

    # Technology & Business
    "Apple reported revenue of $117 billion in Q1 2024.",
    "Tesla delivered 1.8 million vehicles in 2023.",
    "Microsoft acquired Activision Blizzard for $68.7 billion.",

    # Science
    "The human brain contains about 86 billion neurons.",
    "Light travels at approximately 300000 kilometers per second.",
    "The Earth is about 150 million kilometers from the Sun.",

    # Economics
    "India imports approximately 85 percent of its crude oil.",
    "India has a population of 1.4 billion people.",
    "The US national debt exceeds $33 trillion.",
]


# ─────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────

def run_demo(export_csv: bool = True):
    print("\n" + "=" * 70)
    print("  QuantEx — Quantity Extraction Pipeline Demo")
    print("=" * 70)

    all_results = []

    for sentence in SENTENCES:
        print(f"\n📌 {sentence}")
        results = run_pipeline(sentence)

        if results:
            for r in results:
                print(f"   ✅ {r}")
                all_results.append({
                    "sentence": r.sentence,
                    "entity": r.entity,
                    "entity_label": r.entity_label,
                    "value": r.value,
                    "unit": r.unit or "",
                    "unit_type": r.unit_type or "",
                    "context": r.context or "N/A",
                    "raw_quantity": r.raw_quantity,
                    "link_method": r.link_method,
                })
        else:
            print("   ⚠️  No facts extracted")

    # ─────────────────────────────────────────
    # EXPORT TO CSV
    # ─────────────────────────────────────────
    if export_csv and all_results:
        os.makedirs("data", exist_ok=True)
        csv_path = "data/output.csv"
        fields = ["sentence", "entity", "entity_label",
                  "value", "unit", "unit_type",
                  "context", "raw_quantity", "link_method"]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\n{'=' * 70}")
        print(f"  ✅ {len(all_results)} facts extracted from {len(SENTENCES)} sentences")
        print(f"  📄 Results exported to {csv_path}")
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    run_demo()