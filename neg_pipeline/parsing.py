# parsing.py
import spacy
from typing import List, Dict, Any

nlp = spacy.load("en_core_web_sm")
def parse_caption(caption, concepts):
    """
    Enhanced parsing with better attribute-object binding and expanded POS coverage.
    Also extracts (relation, obj1, obj2) in a minimal/compatible way.
    """
    try:
        doc = nlp(caption)
    except Exception as e:
        print(f"Error processing caption: {caption}\n{e}")
        return {
            "objects": [],
            "attributes": {concept: [] for concept in concepts},
            "all_attributes": [],
            "all_objects": [],
            "relations": []
        }
    objects = []
    attributes = {concept: [] for concept in concepts}
    all_attributes = []
    all_objects = []
    relations = []

    # Extract nouns
    for token in doc:
        if token.pos_ == "NOUN":
            objects.append(token.text)
            all_objects.append(token.text)

    # Extract attributes from multiple POS tags
    for token in doc:
        target_noun = None
        attribute_text = token.text

        # 1. ADJECTIVES (including comparatives)
        if token.pos_ == "ADJ":
            all_attributes.append(token.text)

            if token.dep_ == "amod" and token.head.pos_ == "NOUN":
                target_noun = token.head.text
            elif token.dep_ == "acomp":
                for child in token.head.children:
                    if child.dep_ == "nsubj" and child.pos_ == "NOUN":
                        target_noun = child.text
                        break
            elif token.dep_ == "advmod" and token.head.pos_ == "NOUN":
                target_noun = token.head.text

            for concept in concepts:
                attributes[concept].append((attribute_text, target_noun))

            # Comparative relation extraction (additions)
            if hasattr(token, 'morph'):
                degree = token.morph.get("Degree")
                if degree and ("Cmp" in degree or "Sup" in degree):
                    for child in token.children:
                        if child.lemma_.lower() == "than":
                            left = None
                            right = None
                            for ch in token.head.children:
                                if ch.dep_ == "nsubj" and ch.pos_ == "NOUN":
                                    left = ch.text
                                    break
                            for th in child.children:
                                if th.pos_ == "NOUN":
                                    right = th.text
                                    break
                            if left and right:
                                relations.append((token.text, left, right))

        # 2. ADVERBS (orientation, manner, speed, etc.)
        elif token.pos_ == "ADV":
            all_attributes.append(token.text)
            if token.dep_ == "advmod":
                if token.head.pos_ == "VERB":
                    for child in token.head.children:
                        if child.dep_ == "nsubj" and child.pos_ == "NOUN":
                            target_noun = child.text
                            break
                elif token.head.pos_ == "ADJ":
                    if token.head.dep_ == "amod" and token.head.head.pos_ == "NOUN":
                        target_noun = token.head.head.text
                elif token.head.pos_ == "ADV":
                    target_noun = None
            for concept in concepts:
                attributes[concept].append((attribute_text, target_noun))

        # 3. PREPOSITIONS (spatial relations) -- ADD RELATIONS
        elif token.pos_ == "ADP":
            all_attributes.append(token.text)
            # Look for object of the preposition
            right_noun = None
            for child in token.children:
                if child.dep_ in ("pobj", "dobj") and child.pos_ == "NOUN":
                    right_noun = child.text
                    break
            # Try to find left noun (head or ancestor)
            left_noun = None
            left = token.head
            if left.pos_ == "NOUN":
                left_noun = left.text
            else:
                for anc in token.ancestors:
                    if anc.pos_ == "NOUN":
                        left_noun = anc.text
                        break
            # Add relation if both
            if left_noun and right_noun and left_noun != right_noun:
                relations.append((token.text, left_noun, right_noun))
            # Store for all concepts
            for concept in concepts:
                attributes[concept].append((attribute_text, right_noun))

        # 4. COMPARATIVE AND SUPERLATIVE HANDLING (already handled above)
        if token.pos_ == "ADJ" and hasattr(token, 'morph'):
            degree = token.morph.get("Degree")
            if degree and ("Cmp" in degree or "Sup" in degree):
                all_attributes.append(token.text)
                if "Cmp" in degree:
                    than_token = None
                    for child in token.children:
                        if child.lemma_.lower() == "than":
                            than_token = child
                            break
                    if than_token:
                        for child in than_token.children:
                            if child.pos_ == "NOUN":
                                target_noun = f"{token.head.text}_vs_{child.text}" if token.head.pos_ == "NOUN" else child.text
                                break
                for concept in concepts:
                    attributes[concept].append((attribute_text, target_noun))

        # 5. PARTICIPLES (can act as adjectives)
        elif token.pos_ == "VERB" and token.dep_ in ("amod", "acl"):
            all_attributes.append(token.text)
            if token.dep_ == "amod" and token.head.pos_ == "NOUN":
                target_noun = token.head.text
            elif token.dep_ == "acl":
                for child in token.head.children:
                    if child.dep_ == "det" and token.head.pos_ == "NOUN":
                        target_noun = token.head.text
                        break
            for concept in concepts:
                attributes[concept].append((attribute_text, target_noun))

        # 6. SPECIALIZED SPATIAL NOUNS (add relation if in prepositional context)
        elif token.pos_ == "NOUN" and token.lemma_.lower() in [
            "top", "bottom", "side", "front", "back", "center", "middle", 
            "left", "right", "corner", "edge", "surface", "interior", "exterior"
        ]:
            all_attributes.append(token.text)
            for ancestor in token.ancestors:
                if ancestor.pos_ == "ADP":
                    left_noun = None
                    for anc2 in ancestor.ancestors:
                        if anc2.pos_ == "NOUN" and anc2 != token:
                            left_noun = anc2.text
                            break
                    if ancestor.head.pos_ == "NOUN" and ancestor.head != token:
                        left_noun = ancestor.head.text
                    if left_noun:
                        relations.append((ancestor.text, left_noun, token.text))
                    break
            for concept in concepts:
                attributes[concept].append((attribute_text, None))

        # 7. NUMBERS (cardinal quantities)
        elif ("number" in concepts or "any" in concepts) and (token.pos_ == "NUM" or token.like_num):
            all_attributes.append(token.text)
            target_noun = None
            if token.i + 1 < len(doc) and doc[token.i + 1].pos_ == "NOUN":
                target_noun = doc[token.i + 1].text
            else:
                for right in token.rights:
                    if right.pos_ == "NOUN":
                        target_noun = right.text
                        break
            for concept in concepts:
                attributes[concept].append((token.text, target_noun))

    return {
        "objects": objects,
        "attributes": attributes,
        "all_attributes": all_attributes,
        "all_objects": all_objects,
        "relations": relations
    }
