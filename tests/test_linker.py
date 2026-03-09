import sys
sys.path.insert(0, '.')

from quantex.linker import link_quantities_to_entities

sentences = [
    "Virat Kohli scored 183 runs against Pakistan.",
    "The Burj Khalifa stands at 828 meters tall.",
    "Apple reported revenue of $117 billion in Q1 2024.",
    "India imports approximately 85 percent of its crude oil.",
    "Tesla delivered 1.8 million vehicles in 2023.",
    "The human brain contains about 86 billion neurons.",
    "Mount Everest is 8849 meters above sea level.",
]

for sent in sentences:
    print(f"\nINPUT : {sent}")
    facts = link_quantities_to_entities(sent)
    if facts:
        for f in facts:
            print(f"  → {f.entity} ({f.entity_label}) | {f.quantity_value} {f.quantity_unit} | via {f.link_method}")
    else:
        print("  → No facts extracted")