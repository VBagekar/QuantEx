"""
linker.py

Links each detected QuantitySpan to the most relevant EntitySpan.

Core idea: In a dependency parse tree, the entity that "owns" a quantity
is the one connected to it via the shortest path in the tree.

Example:
    "Virat Kohli scored 183 runs against Pakistan"
    
    Dependency tree:
        scored (ROOT)
        ├── Kohli (nsubj)    ← PERSON entity
        ├── runs (dobj)
        │   └── 183 (nummod) ← quantity token
        └── Pakistan (prep→pobj) ← GPE entity
    
    For quantity 183:
        path to Kohli:    183 → runs → scored → Kohli  (length 3)
        path to Pakistan: 183 → runs → scored → against → Pakistan (length 4)
    
    Winner: Kohli (shorter path) ✓

Output: LinkedFact(entity="Virat Kohli", quantity=183, unit="runs", ...)
"""

import spacy
from dataclasses import dataclass
from typing import Optional

from quantex.quantity_detector import QuantitySpan, detect_quantities
from quantex.entity_extractor import EntitySpan, extract_entities, nlp


# ─────────────────────────────────────────────
# 1. DATA CLASS — Final output of the whole pipeline
# ─────────────────────────────────────────────

@dataclass
class LinkedFact:
    entity: str               # e.g., "Virat Kohli"
    entity_label: str         # e.g., "PERSON"
    quantity_value: float     # e.g., 183.0
    quantity_unit: Optional[str]      # e.g., "run"
    quantity_unit_type: Optional[str] # e.g., "count"
    raw_quantity: str         # e.g., "183 runs"
    sentence: str             # original sentence
    link_method: str          # "dep_tree" or "char_distance"


# ─────────────────────────────────────────────
# 2. DEPENDENCY TREE PATH LENGTH
#
# To find how "close" a quantity token is to an entity in the
# dependency tree, we walk UP the tree from each token until
# we find their common ancestor, counting steps.
#
# This is essentially finding the LCA (Lowest Common Ancestor)
# in the dependency tree and measuring path length through it.
# ─────────────────────────────────────────────

def _get_ancestors(token) -> list:
    """
    Returns the list of ancestor tokens from token → ROOT.
    Example: "183" → [runs, scored] (ROOT)
    """
    ancestors = []
    current = token
    while current.head != current:  # stop at ROOT (ROOT's head is itself)
        current = current.head
        ancestors.append(current)
    return ancestors


def _dep_tree_distance(token_a, token_b) -> int:
    """
    Compute the shortest path length between two tokens
    in the dependency tree.

    Algorithm:
    1. Get ancestor chain for both tokens
    2. Find lowest common ancestor (LCA)
    3. Distance = steps_from_a_to_LCA + steps_from_b_to_LCA
    """
    ancestors_a = _get_ancestors(token_a)
    ancestors_b = _get_ancestors(token_b)

    # Add the tokens themselves to their ancestor lists
    chain_a = [token_a] + ancestors_a
    chain_b = [token_b] + ancestors_b

    # Find the first common token (LCA)
    set_b = {t.i for t in chain_b}  # use token index for comparison
    for i, tok in enumerate(chain_a):
        if tok.i in set_b:
            lca = tok
            j = next(k for k, t in enumerate(chain_b) if t.i == lca.i)
            return i + j  # total path length through LCA

    return 999  # no common ancestor found (shouldn't happen in valid parse)


# ─────────────────────────────────────────────
# 3. FIND QUANTITY TOKEN IN DOC
#
# Our QuantitySpan has char_start/char_end positions.
# We need to find the actual spaCy token at that position
# so we can compute dependency tree distances.
# ─────────────────────────────────────────────

def _find_quantity_token(doc, quantity: QuantitySpan):
    """
    Find the numeric token in the doc that corresponds to a QuantitySpan.
    We match by character position.
    """
    for token in doc:
        # The number token should start within the quantity's span
        if token.idx >= quantity.char_start and token.idx < quantity.char_end:
            if token.like_num or token.is_digit:
                return token
    return None


# ─────────────────────────────────────────────
# 4. FIND ENTITY TOKEN IN DOC
# ─────────────────────────────────────────────

def _find_entity_token(doc, entity: EntitySpan):
    """
    Find the head token of an entity span in the doc.
    For multi-word entities like "Virat Kohli", we return
    the syntactic head (usually the last token = "Kohli").
    """
    entity_tokens = [
        t for t in doc
        if t.idx >= entity.char_start and t.idx < entity.char_end
    ]
    if not entity_tokens:
        return None
    # Return the token with no parent inside the span (syntactic head)
    for t in entity_tokens:
        if t.head not in entity_tokens:
            return t
    return entity_tokens[-1]


# ─────────────────────────────────────────────
# 5. LINK BY DEPENDENCY TREE
# ─────────────────────────────────────────────

def _link_by_dep_tree(doc, quantity: QuantitySpan,
                       entities: list[EntitySpan]) -> tuple[Optional[EntitySpan], int]:
    """
    Find the entity with the shortest dependency path to the quantity.
    Returns (best_entity, distance).
    """
    qty_token = _find_quantity_token(doc, quantity)
    if qty_token is None:
        return None, 999

    best_entity = None
    best_distance = 999

    for entity in entities:
        ent_token = _find_entity_token(doc, entity)
        if ent_token is None:
            continue
        dist = _dep_tree_distance(qty_token, ent_token)
        if dist < best_distance:
            best_distance = dist
            best_entity = entity

    return best_entity, best_distance


# ─────────────────────────────────────────────
# 6. LINK BY CHARACTER DISTANCE (fallback)
#
# If dependency linking fails, just find the entity
# whose midpoint in the text is closest to the quantity's midpoint.
# ─────────────────────────────────────────────

def _link_by_char_distance(quantity: QuantitySpan,
                            entities: list[EntitySpan]) -> Optional[EntitySpan]:
    """
    Find entity closest to quantity by character position.
    """
    qty_mid = (quantity.char_start + quantity.char_end) / 2

    best_entity = None
    best_dist = float("inf")

    for entity in entities:
        ent_mid = (entity.char_start + entity.char_end) / 2
        dist = abs(qty_mid - ent_mid)
        if dist < best_dist:
            best_dist = dist
            best_entity = entity

    return best_entity


# ─────────────────────────────────────────────
# 7. MAIN FUNCTION
# ─────────────────────────────────────────────

def link_quantities_to_entities(sentence: str) -> list[LinkedFact]:
    """
    Full linking pipeline for a single sentence.

    Steps:
    1. Detect all quantities in sentence
    2. Extract all entities in sentence
    3. For each quantity, find the best matching entity
    4. Return list of LinkedFact objects

    Example:
        link_quantities_to_entities("Virat Kohli scored 183 runs against Pakistan")
        → [LinkedFact(entity="Virat Kohli", quantity_value=183, unit="run", ...)]
    """
    quantities = detect_quantities(sentence)
    entities = extract_entities(sentence)

    if not quantities or not entities:
        return []

    doc = nlp(sentence)
    facts = []

    for qty in quantities:
        # Try dependency tree linking first
        best_entity, dep_dist = _link_by_dep_tree(doc, qty, entities)

        if best_entity and dep_dist < 10:
            method = "dep_tree"
        else:
            # Fall back to character distance
            best_entity = _link_by_char_distance(qty, entities)
            method = "char_distance"

        if best_entity:
            facts.append(LinkedFact(
                entity=best_entity.text,
                entity_label=best_entity.label,
                quantity_value=qty.value,
                quantity_unit=qty.unit,
                quantity_unit_type=qty.unit_type,
                raw_quantity=qty.raw_text,
                sentence=sentence,
                link_method=method,
            ))

    return facts