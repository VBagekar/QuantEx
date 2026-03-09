import sys
sys.path.insert(0, '.')

from quantex.quantity_detector import detect_quantities
from quantex.quantity_detector import detect_quantities

sentences = [
    "India has a population of 1.4 billion people.",
    "Virat Kohli scored 183 runs against Pakistan.",
    "The Burj Khalifa stands at 828 meters tall.",
    "Apple reported revenue of $117 billion in Q1 2024.",
    "India imports approximately 85 percent of its crude oil.",
    "The human brain contains about 86 billion neurons.",
]

for sent in sentences:
    print(f"\nINPUT : {sent}")
    results = detect_quantities(sent)
    for q in results:
        print(f"  → value={q.value}, unit={q.unit}, type={q.unit_type}, raw='{q.raw_text}'")