"""Checkable facts inside a claim: numbers, named entities and negation.

Embeddings say whether a claim and a sentence are *about the same thing*. They do not notice
that "45 kilograms" is not "21.5 kilograms". These checks do.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Entity types that carry checkable facts. CARDINAL / PERCENT / MONEY are handled as numbers.
FACT_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "DATE", "NORP", "FAC", "EVENT", "PRODUCT", "WORK_OF_ART", "LAW"}
WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20, "hundred": 100,
    "thousand": 1000, "million": 1e6, "billion": 1e9, "half": 0.5, "dozen": 12,
}
_NUM_RE = re.compile(r"(?<![\w.])\$?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)")


@dataclass
class Facts:
    numbers: set[float]
    entities: list[tuple[str, str]]     # (text, spaCy label)


def numbers(text: str) -> set[float]:
    """Numeric values in the text: '1,200' -> 1200, '12%' -> 12, '$1.6 billion' -> 1.6 and 1e9, 'two' -> 2."""
    nums = {float(m.group(1).replace(",", "")) for m in _NUM_RE.finditer(text)}
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in WORD_NUMBERS:
            nums.add(float(WORD_NUMBERS[word]))
    return nums


def has_number(pool: set[float], n: float) -> bool:
    return any(abs(p - n) < 1e-9 for p in pool)


def fmt_numbers(nums: set[float]) -> str:
    return ", ".join(f"{n:g}" for n in sorted(nums))


def norm_entity(ent: str) -> str:
    return re.sub(r"^(the|a|an)\s+", "", ent.lower()).strip(" .,'\"")


def extract(nlp, text: str) -> Facts:
    doc = nlp(text)
    # Relative dates without numbers ("last year", "the century") are too paraphrase-prone to check.
    ents = [(e.text, e.label_) for e in doc.ents
            if e.label_ in FACT_ENTITY_LABELS and not (e.label_ == "DATE" and not numbers(e.text))]
    return Facts(numbers(text), ents)


def conflicts(nlp, claim_text: str, claim: Facts, related: list[str], source_lower: str) -> list[str]:
    """Ways the claim disagrees with closely related evidence (related[0] is the best sentence)."""
    ev_text = " ".join(related)
    ev = extract(nlp, ev_text)
    out = []

    # Numbers are compared with the single best-matching sentence: it states the same fact.
    best_nums = numbers(related[0]) or ev.numbers
    missing = {n for n in claim.numbers if not has_number(best_nums, n)}
    if missing and best_nums:
        out.append(f"number mismatch: claim says {fmt_numbers(missing)}, evidence says {fmt_numbers(best_nums)}")

    ev_lower = ev_text.lower()
    claim_ent_text = " ".join(norm_entity(e) for e, _ in claim.entities)
    for ent, label in claim.entities:
        # an entity that appears anywhere in the sources is grounded, so it is not a conflict
        if norm_entity(ent) in source_lower or norm_entity(ent) in ev_lower or \
                (numbers(ent) and numbers(ent) <= ev.numbers):
            continue
        rivals = [t for t, l in ev.entities if l == label and norm_entity(t) not in claim_ent_text]
        if rivals:
            # prefer the rival that shares a word with the claim's entity ("Atlantic Ocean" -> "Pacific Ocean")
            words = set(norm_entity(ent).split())
            rival = max(rivals, key=lambda r: len(words & set(norm_entity(r).split())))
            out.append(f"{label} mismatch: claim says '{ent}', evidence says '{rival}'")

    if polarity_flip(nlp, claim_text, related[0]):
        out.append("polarity mismatch: negation differs from evidence")
    return out


def polarity_flip(nlp, claim: str, evidence: str) -> bool:
    """Negation attached to a verb that both texts share ('was not compatible' vs 'was compatible')."""
    c, e = nlp(claim), nlp(evidence)

    def negated_heads(doc):
        return {t.head.lemma_.lower() for t in doc if t.dep_ == "neg"}

    c_lemmas, e_lemmas = {t.lemma_.lower() for t in c}, {t.lemma_.lower() for t in e}
    c_neg, e_neg = negated_heads(c), negated_heads(e)
    return bool((e_neg - c_neg) & c_lemmas) or bool((c_neg - e_neg) & e_lemmas)


def novel(claim: Facts, source_lower: str) -> list[str]:
    """Numbers and names in the claim that appear in no source at all."""
    src_nums = numbers(source_lower)
    out = [fmt_numbers({n}) for n in claim.numbers if not has_number(src_nums, n)]
    for ent, _ in claim.entities:
        if norm_entity(ent) not in source_lower and not (numbers(ent) and numbers(ent) <= src_nums):
            out.append(f"'{ent}'")
    return out
