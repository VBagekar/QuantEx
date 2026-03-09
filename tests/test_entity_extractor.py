import sys
sys.path.insert(0, '.')

from quantex.entity_extractor import extract_entities

sentences = [
    "Virat Kohli scored 183 runs against Pakistan.",
    "The Burj Khalifa stands at 828 meters tall.",
    "Apple reported revenue of $117 billion in Q1 2024.",
    "India imports approximately 85 percent of its crude oil.",
    "Tesla delivered 1.8 million vehicles in 2023.",
    "The Amazon River stretches approximately 6400 kilometers.",
]

for sent in sentences:
    print(f"\nINPUT : {sent}")
    entities = extract_entities(sent)
    for e in entities:
        print(f"  → '{e.text}' | label={e.label} | source={e.source}")