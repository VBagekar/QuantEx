"""
entity_extractor.py

Extracts named entities from text using two complementary strategies:

Strategy 1 — spaCy NER:
    Finds named entities like PERSON, ORG, GPE (geo-political entities like
    countries/cities), FAC (facilities like buildings), and NORP (nationalities).
    Example: "Virat Kohli" → PERSON, "India" → GPE, "Tesla" → ORG

Strategy 2 — Dependency Parsing (fallback):
    If NER misses something, we look at the grammatical structure of the sentence.
    The "nsubj" (nominal subject) of the main verb is almost always the entity
    the sentence is talking about.
    Example: "The Burj Khalifa stands at 828 meters"
             nsubj of "stands" → "Burj Khalifa"

Why two strategies?
    NER is trained on news/Wikipedia text and can miss domain-specific entities.
    Dependency parsing is rule-based and catches the grammatical subject reliably.
    Together they cover almost all factual sentences.
"""

import spacy
from dataclasses import dataclass

# ─────────────────────────────────────────────
# 1. DATA CLASS
# ─────────────────────────────────────────────

@dataclass
class EntitySpan:
    text: str           # e.g., "Virat Kohli"
    label: str          # e.g., "PERSON", "ORG", "GPE"
    source: str         # "ner" or "dep" — which strategy found it
    char_start: int     # position in sentence (used for linking to quantities)
    char_end: int


# ─────────────────────────────────────────────
# 2. ENTITY TYPES WE CARE ABOUT
#
# spaCy has 18+ entity types. We only want entities that make sense
# as the "subject" of a quantity fact.
# e.g., we want PERSON ("Kohli scored 183 runs") but NOT DATE, TIME, CARDINAL
# ─────────────────────────────────────────────

RELEVANT_LABELS = {
    "PERSON",   # Virat Kohli, Elon Musk
    "ORG",      # Tesla, Apple, ISRO
    "GPE",      # India, USA, Mumbai (Geo-Political Entity)
    "LOC",      # Amazon River, Mount Everest (non-GPE locations)
    "FAC",      # Burj Khalifa, Golden Gate Bridge (facilities/landmarks)
    "NORP",     # Indians, Americans (nationalities/groups)
    "PRODUCT",  # iPhone, Model S
}

# Dependency relations that indicate the main subject of a sentence
SUBJECT_DEPS = {"nsubj", "nsubjpass"}


# ─────────────────────────────────────────────
# 3. LOAD spaCy MODEL
#
# en_core_web_trf = transformer-based model (most accurate)
# We load it once at module level so it's not reloaded on every call.
# If trf isn't available, fallback to en_core_web_lg.
# ─────────────────────────────────────────────

def _load_model():
    try:
        return spacy.load("en_core_web_trf")
    except OSError:
        try:
            return spacy.load("en_core_web_lg")
        except OSError:
            raise OSError(
                "No spaCy model found. Run: python -m spacy download en_core_web_trf"
            )

nlp = _load_model()


# ─────────────────────────────────────────────
# 4. STRATEGY 1 — NER-based extraction
# ─────────────────────────────────────────────

def _extract_by_ner(doc) -> list[EntitySpan]:
    """
    Use spaCy's built-in NER to find named entities.
    Filter to only keep entity types in RELEVANT_LABELS.
    """
    entities = []
    for ent in doc.ents:
        if ent.label_ in RELEVANT_LABELS:
            entities.append(EntitySpan(
                text=ent.text,
                label=ent.label_,
                source="ner",
                char_start=ent.start_char,
                char_end=ent.end_char,
            ))
    return entities


# ─────────────────────────────────────────────
# 5. STRATEGY 2 — Dependency parsing fallback
#
# How dependency parsing works:
# Every sentence has a ROOT verb. Nouns connected to the ROOT
# with an "nsubj" relation are the grammatical subjects.
#
# Example parse of "The Burj Khalifa stands at 828 meters":
#   stands (ROOT)
#   ├── Khalifa (nsubj)  ← this is our entity
#   │   └── Burj (compound)
#   │   └── The (det)
#   └── at (prep)
#       └── meters (pobj)
#           └── 828 (nummod)
#
# We collect the nsubj token AND all its compound children
# to reconstruct "Burj Khalifa" (not just "Khalifa").
# ─────────────────────────────────────────────

def _get_compound_span(token) -> tuple[str, int, int]:
    """
    Given a token, collect it plus all 'compound' children
    to reconstruct the full name.

    e.g., token = "Khalifa", compounds = ["Burj"] → "Burj Khalifa"
    """
    tokens = [token]
    for child in token.children:
        if child.dep_ == "compound":
            tokens.append(child)

    tokens.sort(key=lambda t: t.i)  # sort by position in sentence
    full_text = " ".join(t.text for t in tokens)
    start = tokens[0].idx
    end = tokens[-1].idx + len(tokens[-1].text)
    return full_text, start, end


def _extract_by_dep(doc) -> list[EntitySpan]:
    """
    Find nominal subjects (nsubj) from the dependency tree.
    Only extract noun/proper noun tokens.
    """
    entities = []
    for token in doc:
        if token.dep_ in SUBJECT_DEPS and token.pos_ in {"NOUN", "PROPN"}:
            full_text, start, end = _get_compound_span(token)
            entities.append(EntitySpan(
                text=full_text,
                label="SUBJECT",  # we don't know exact type, label as SUBJECT
                source="dep",
                char_start=start,
                char_end=end,
            ))
    return entities


# ─────────────────────────────────────────────
# 6. DEDUPLICATION
#
# Both strategies might find the same entity.
# e.g., NER finds "India" (GPE) and dep parsing also finds "India" (SUBJECT)
# We keep the NER version (more informative label) and drop duplicates.
# ─────────────────────────────────────────────

def _deduplicate(entities: list[EntitySpan]) -> list[EntitySpan]:
    """
    Remove duplicate entities based on character span overlap.
    Prefer NER entities over dep-parsed ones.
    """
    # Sort: NER first, dep second
    entities.sort(key=lambda e: 0 if e.source == "ner" else 1)

    seen_spans = []
    unique = []
    for ent in entities:
        # Check if this span overlaps with any already-kept entity
        overlaps = any(
            ent.char_start < seen_end and ent.char_end > seen_start
            for seen_start, seen_end in seen_spans
        )
        if not overlaps:
            unique.append(ent)
            seen_spans.append((ent.char_start, ent.char_end))

    return unique


# ─────────────────────────────────────────────
# 7. MAIN FUNCTION
# ─────────────────────────────────────────────

def extract_entities(text: str) -> list[EntitySpan]:
    """
    Extract entities from text using NER + dependency parsing.

    Example:
        extract_entities("Virat Kohli scored 183 runs against Pakistan")
        → [EntitySpan("Virat Kohli", "PERSON", "ner", 0, 11),
           EntitySpan("Pakistan", "GPE", "ner", 31, 39)]
    """
    doc = nlp(text)

    ner_entities = _extract_by_ner(doc)
    dep_entities = _extract_by_dep(doc)

    all_entities = ner_entities + dep_entities
    return _deduplicate(all_entities)