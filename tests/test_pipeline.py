import sys
sys.path.insert(0, '.')

from quantex.pipeline import run_pipeline

sentences = [
    "Virat Kohli scored 183 runs against Pakistan.",
    "The Burj Khalifa stands at 828 meters tall.",
    "Apple reported revenue of $117 billion in Q1 2024.",
    "India imports approximately 85 percent of its crude oil.",
    "Tesla delivered 1.8 million vehicles in 2023.",
    "The human brain contains about 86 billion neurons.",
    "Mount Everest is 8849 meters above sea level.",
    "India has a population of 1.4 billion people.",
]

print("=" * 65)
print("QuantEx Pipeline — Full Output")
print("=" * 65)

for sent in sentences:
    print(f"\nINPUT : {sent}")
    results = run_pipeline(sent)
    if results:
        for r in results:
            print(f"  {r}")
    else:
        print("  → No facts extracted")