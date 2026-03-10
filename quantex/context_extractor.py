"""
context_extractor.py

Extracts the contextual predicate connecting an entity to a quantity.

In the sentence "Apple reported revenue of $117 billion":
    - Entity:   Apple
    - Quantity: $117 billion  
    - Context:  "revenue" (the noun bridging entity → quantity)
               OR "reported" (the verb)

We extract context by finding the lowest common ancestor (LCA)
in the dependency tree between the entity token and quantity token.
The LCA or its direct dependent is almost always the context word.

Example dependency tree:
    reported (ROOT)
    ├── Apple (nsubj)        ← entity
    └── revenue (dobj)
        └── of (prep)
            └── billion (pobj)
                └── 117 (nummod)  ← quantity

LCA of "Apple" and "117" = "reported"
But "revenue" is the direct object — more meaningful as context.
So we take the LCA's most informative dependent.
"""

import spacy
from typing import Optional
from quantex.entity_extractor import nlp


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

SKIP_VERBS = {"be", "have", "do", "stand", "remain", "become", "seem"}

SCALE_WORDS = {"million", "billion", "trillion", "thousand", "lakh", "crore"}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _ancestors(token) -> list:
    chain = []
    cur = token
    while cur.head != cur:
        cur = cur.head
        chain.append(cur)
    return chain


def _lca(token_a, token_b):
    """Find lowest common ancestor of two tokens in dep tree."""
    ancestors_a = [token_a] + _ancestors(token_a)
    set_a = {t.i for t in ancestors_a}

    cur = token_b
    while cur.head != cur:
        if cur.i in set_a:
            return cur
        cur = cur.head
    return cur  # ROOT


# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────

def extract_context(sentence: str,
                    entity_text: str,
                    quantity_raw: str,
                    unit: Optional[str] = None) -> Optional[str]:
    """
    Extract the contextual word linking entity to quantity.
    Skips generic verbs, unit words, scale words, and entity words.
    """
    doc = nlp(sentence)

    # Find entity head token
    entity_token = None
    entity_lower = entity_text.lower()
    for token in doc:
        if token.text.lower() in entity_lower or entity_lower in token.text.lower():
            entity_token = token
            break

    # Find quantity token (numeric)
    qty_token = None
    for token in doc:
        if token.like_num or token.is_digit:
            qty_token = token
            break

    if entity_token is None or qty_token is None:
        return None

    # ─────────────────────────────────────────
    # BUILD SKIP SET
    # Includes: scale words, unit words (both norm + surface forms),
    # entity words — so we never return any of these as context
    # ─────────────────────────────────────────
    skip_lemmas = set(SCALE_WORDS)  # always skip scale words

    # Add unit words from the passed unit string (e.g. "million vehicle")
    if unit:
        for w in unit.split():
            skip_lemmas.add(w.strip().lower())

    # Add direct head lemma of qty_token (the unit noun in the sentence)
    unit_lemma = qty_token.head.lemma_.lower() if qty_token.head != qty_token else None
    if unit_lemma:
        skip_lemmas.add(unit_lemma)

    # Walk up 3 levels from qty_token — only skip NOUNS and SCALE words
    # Never add verbs, or we'll accidentally block the context itself
    cur = qty_token
    for _ in range(3):
        if cur.head == cur:
            break
        if cur.head.pos_ in {"NOUN", "PROPN"} or cur.head.lemma_.lower() in SCALE_WORDS:
            skip_lemmas.add(cur.head.text.lower())
            skip_lemmas.add(cur.head.lemma_.lower())
        cur = cur.head

    # Never return the entity itself as context e.g. "brain" for brain entity
    for word in entity_text.lower().split():
        skip_lemmas.add(word)

    # ─────────────────────────────────────────
    # FIND LCA
    # ─────────────────────────────────────────
    lca = _lca(entity_token, qty_token)

    def is_good_context(token) -> bool:
        """A context word must not be a unit, scale, entity, or generic verb."""
        lemma = token.lemma_.lower()
        if lemma in skip_lemmas:
            return False
        if token.pos_ == "VERB" and lemma in SKIP_VERBS:
            return False
        if token.pos_ in {"DET", "ADP", "PUNCT", "SPACE", "NUM"}:
            return False
        return True

    # ─────────────────────────────────────────
    # STRATEGY 1: LCA is a meaningful verb
    # First look for a noun child (more informative than verb itself)
    # e.g. "reported" → look for "revenue" as dobj
    # ─────────────────────────────────────────
    if lca.pos_ == "VERB" and is_good_context(lca):
        for child in lca.children:
            if child.pos_ == "NOUN" and child.dep_ in {"dobj", "attr", "nsubj", "pobj"}:
                if is_good_context(child):
                    return child.lemma_
        return lca.lemma_

    # ─────────────────────────────────────────
    # STRATEGY 2: LCA is a meaningful noun
    # e.g. "population" in "India has a population of 1.4 billion"
    # ─────────────────────────────────────────
    if lca.pos_ == "NOUN" and is_good_context(lca):
        return lca.lemma_

    # ─────────────────────────────────────────
    # STRATEGY 3: Walk UP from quantity token
    # Skips unit nouns and scale words via is_good_context
    # e.g. 1.8 → vehicle (skip) → deliver ✓
    #      86  → neuron (skip)  → contain ✓
    # ─────────────────────────────────────────
    cur = qty_token
    visited = set()
    while cur.head != cur and cur.i not in visited:
        visited.add(cur.i)
        cur = cur.head
        if is_good_context(cur):
            if cur.pos_ == "VERB":
                # Check for a better noun child first
                for child in cur.children:
                    if child.pos_ == "NOUN" and child.dep_ in {"dobj", "attr", "pobj"}:
                        if is_good_context(child):
                            return child.lemma_
                return cur.lemma_
            if cur.pos_ == "NOUN":
                return cur.lemma_

    # ─────────────────────────────────────────
    # STRATEGY 4: Walk UP from entity token
    # Last resort — find any good verb above the entity
    # ─────────────────────────────────────────
    cur = entity_token
    visited = set()
    while cur.head != cur and cur.i not in visited:
        visited.add(cur.i)
        cur = cur.head
        if is_good_context(cur) and cur.pos_ == "VERB":
            return cur.lemma_

    return None