# QuantEx — Quantity Extractor from Text

A structured information extraction pipeline that identifies and links **quantities** to their associated **entities** and **context** from raw text.

## Example
Input:
> "India has a population of 1.4 billion people and a GDP of $3.7 trillion."

Output:
| Entity | Quantity | Unit    | Context     |
|--------|----------|---------|-------------|
| India  | 1.4      | billion | population  |
| India  | 3.7      | trillion dollars | GDP |

## Pipeline
1. **Quantity Detection** — regex + spaCy to find numeric expressions
2. **Entity Extraction** — NER + dependency parsing
3. **Entity-Quantity Linking** — dependency tree traversal
4. **Context Extraction** — surrounding predicate/noun context

## Setup
pip install -r requirements.txt
python -m spacy download en_core_web_trf

## Usage
python demo.py

## Motivation
Inspired by [QuTE (SIGMOD 2021)](https://dl.acm.org/doi/10.1145/3448016.3452791) and contextualized quantity fact extraction work by Dr. Koninika Pal et al.
```

---

### Step 5 — Fill `data/sample_sentences.txt`

Paste this test data we'll use throughout:
```
India has a population of 1.4 billion people.
Virat Kohli scored 183 runs against Pakistan in a test match.
The Burj Khalifa stands at 828 meters tall.
Apple reported revenue of $117 billion in Q1 2024.
A blue whale can weigh up to 200 tonnes.
The Amazon River stretches approximately 6400 kilometers.
Tesla delivered 1.8 million vehicles in 2023.
Mount Everest is 8849 meters above sea level.
India imports approximately 85 percent of its crude oil requirements.
The human brain contains about 86 billion neurons.