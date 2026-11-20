"""Checkable facts inside a claim: numbers, named entities and negation.

Embeddings say whether a claim and a sentence are *about the same thing*. They do not notice
that "45 kilograms" is not "21.5 kilograms". These checks do.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Entity types that carry checkable facts. CARDINAL / PERCENT / MONEY are handled as numbers.
FACT_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "DATE", "NORP", "FAC", "EVENT", "PRODUCT", "WORK_OF_ART", "LAW"}
_UNITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
         "ninety": 90}
_ORDINAL_UNITS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
                  "eighth": 8, "ninth": 9}
_SCALES = {"hundred": 100, "thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12,
           "k": 1e3, "m": 1e6, "b": 1e9, "bn": 1e9}
# "second" is left out: it is usually the time unit ("a second", "30 seconds"); see _word_value.
WORD_NUMBERS = {
    **_UNITS, **_TENS, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "hundred": 100, "thousand": 1000,
    "million": 1e6, "billion": 1e9, "trillion": 1e12, "half": 0.5, "dozen": 12,
    "first": 1, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
    "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20,
}
# A numeral with an optional sign, currency, decimal part and scale: '-40', '$1,234.56', '$1.6 billion', '1.2M'.
# The sign only counts when nothing word-like is glued to its left, so '1990-1995' and 'COVID-19' stay positive.
# Lower-case k/m/b only scale after a currency sign, so '5 m' (meters) is not 5 million.
_NUM_RE = re.compile(
    r"(?<![\w.])(?P<sign>-)?(?P<cur>[$\u20ac\u00a3])?(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"(?:\s?(?P<scale>(?i:thousand|million|billion|trillion))\b"
    r"|(?P<abbr>(?(cur)(?i:bn|k|m|b)|(?:bn|K|M|B)))\b"
    r"|(?:st|nd|rd|th)\b)?")
_TRANS = str.maketrans({"\u2018": "'", "\u2019": "'", "\u02bc": "'", "\u201c": '"', "\u201d": '"',
                        "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00a0": " "})
# Words after a number that say nothing about what it counts ("18 to 24", "in 1969 the").
_STOP = {"a", "an", "the", "of", "to", "in", "on", "at", "for", "from", "by", "with", "and", "or", "but", "as",
         "is", "was", "are", "were", "be", "been", "has", "had", "have", "than", "that", "which", "who", "per",
         "about", "over", "under", "after", "before", "while", "when", "more", "less", "nor", "so", "it", "its",
         "this", "these", "those", "also", "up", "down", "out", "into", "between", "among", "each", "every",
         "other", "another", "same", "such", "s", "nearly", "around", "approximately", "roughly", "almost"}
# Nouns that label the number after them: 'type 1 diabetes', 'phase 3', 'version 2'.
_LABEL_NOUNS = {"type", "phase", "stage", "grade", "level", "class", "category", "version", "chapter", "section",
                "page", "room", "flight", "gate", "route", "step", "tier", "round", "game", "season", "episode"}
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"]
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_CALENDAR = {**{m: i for i, m in enumerate(_MONTHS)}, **{m[:3]: i for i, m in enumerate(_MONTHS)}, "sept": 8,
             **{d: 12 + i for i, d in enumerate(_WEEKDAYS)}}
_CALENDAR_RE = re.compile(r"\b(" + "|".join(sorted(_CALENDAR, key=len, reverse=True)) + r")\b")
# Leading job titles and honorifics before a person's name: 'CFO Desmond Okafor', 'Dr. Halvora Brenquist'.
_ROLE = r"chief(?: [a-z]+){1,2} officer|c[a-z]o|(?:vice )?president|chair(?:man|woman|person)?"
_HONORIFIC = r"dr|mr|mrs|ms|miss|prof|professor|sir|dame|lord|lady|rev|sen|senator|gov|governor|judge"
_TITLES_RE = re.compile(rf"^((?:(?:{_ROLE}|{_HONORIFIC})\.? )+)")
_ROLE_BEFORE_RE = re.compile(rf"(?<![a-z])({_ROLE})\.? $")
_ROLE_ABBR = {"ceo": "chief executive officer", "cfo": "chief financial officer", "coo": "chief operating officer",
              "cto": "chief technology officer", "cio": "chief information officer",
              "cmo": "chief marketing officer", "chair": "chairman", "chairwoman": "chairman",
              "chairperson": "chairman"}


@dataclass
class Facts:
    numbers: set[float]
    entities: list[tuple[str, str]]     # (text, spaCy label)


@dataclass(frozen=True)
class Quantity:
    value: float
    kinds: frozenset[str]     # what the number counts: '%', 'money', 'calendar year', 'kilogram', ... (empty if unknown)


def fold(text: str) -> str:
    """Lower-case text with typographic quotes and dashes made plain, for substring checks."""
    return re.sub(r"\s+", " ", text.translate(_TRANS)).lower()


def numbers(text: str) -> set[float]:
    """Numeric values in the text: '1,200' -> 1200, '12%' -> 12, '$1.6 billion' -> 1.6e9, 'twenty-five' -> 25."""
    return {q.value for q in quantities(text)}


def quantities(text: str) -> list[Quantity]:
    """Every number in the text with what it counts."""
    t = re.sub(r"\s+", " ", text.translate(_TRANS))
    low = t.lower()
    out, masked = [], list(low)
    for m in _NUM_RE.finditer(t):
        out.append(_numeral(m, t, low))
        masked[m.start():m.end()] = " " * (m.end() - m.start())     # 'million' in '$2 million' is not read again
    out += _word_quantities(t, low, "".join(masked))
    return out


def _numeral(m: re.Match, t: str, low: str) -> Quantity:
    value = float(m["num"].replace(",", ""))
    scale = m["scale"] or m["abbr"]
    if scale:
        value *= _SCALES[scale.lower()]
    if m["sign"]:
        value = -value
    is_year = re.fullmatch(r"\d{4}", m["num"]) is not None and 1000 <= value <= 2100 and not (scale or m["sign"])
    return Quantity(value, _kinds(t, low, m.start(), m.end(), bool(m["cur"]), is_year))


def _word_quantities(t: str, low: str, masked: str) -> list[Quantity]:
    """Number words: 'two' -> 2, 'seventy-five' -> 75, 'third' -> 3, 'two hundred' -> 200."""
    words = list(re.finditer(r"[a-z]+", masked))
    out, i = [], 0
    while i < len(words):
        value, j = _word_value(words, i, masked)
        if value is None:
            i += 1
            continue
        while j < len(words) and words[j].group() in WORD_NUMBERS and _SCALES.get(words[j].group(), 0) >= 100 \
                and masked[words[j - 1].end():words[j].start()] == " ":
            value *= _SCALES[words[j].group()]
            j += 1
        out.append(Quantity(value, _kinds(t, low, words[i].start(), words[j - 1].end(), False, False)))
        i = j
    return out


def _word_value(words: list[re.Match], i: int, masked: str) -> tuple[float | None, int]:
    w = words[i].group()
    prev = words[i - 1].group() if i else ""
    nxt = words[i + 1].group() if i + 1 < len(words) else ""
    gap = masked[words[i].end():words[i + 1].start()] if nxt else ""
    if w in _TENS and gap in ("-", " ") and (nxt in _UNITS or nxt in _ORDINAL_UNITS):
        return float(_TENS[w] + (_UNITS.get(nxt) or _ORDINAL_UNITS[nxt])), i + 2
    # pronoun 'one' ('one of the largest', 'no one knows') counts nothing
    if w == "one" and (nxt == "of" or prev in {"no", "the", "any", "every", "each", "which", "this", "that"}):
        return None, i + 1
    if w == "second" and prev in {"the", "its", "his", "her", "their", "our", "my", "your"}:
        return 2.0, i + 1
    if w in WORD_NUMBERS:
        return float(WORD_NUMBERS[w]), i + 1
    return None, i + 1


def _kinds(t: str, low: str, start: int, end: int, money: bool, is_year: bool) -> frozenset[str]:
    """What a number counts, from the text around it."""
    rest = low[end:]
    if money or re.match(r" ?(dollars?|usd|euros?|pounds?)\b", rest):
        return frozenset({"money"})
    if re.match(r" ?(%|percent\b|per cent\b)", rest):
        return frozenset({"%"})
    if re.match(r":\d", rest) or re.search(r"\d:$", low[:start]):
        return frozenset({"time"})
    if is_year:
        return frozenset({"calendar year"})     # not the "year" of "12-year-old"
    kinds = set()
    nxt = re.match(r"[ -]+([a-z]+)", rest)
    if nxt and nxt.group(1) not in _STOP:
        kinds.add(_singular(nxt.group(1)))
    prev = re.search(r"([A-Za-z]+)[ -]*$", t[:start])
    if prev:
        word = prev.group(1).lower()
        if word in _CALENDAR and _CALENDAR[word] < 12:
            kinds.add("day")                                    # 'July 20'
        elif word in _LABEL_NOUNS or (prev.group(1)[0].isupper() and word not in _STOP and not word.endswith("ly")):
            kinds.add(word)                                     # 'Type 1', 'Apollo 11', 'Python 3.0'
    return frozenset(kinds)


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def has_number(pool: set[float], n: float) -> bool:
    # relative tolerance: 2.01 * 1e9 must equal 2,010,000,000
    return any(math.isclose(p, n, rel_tol=1e-9, abs_tol=1e-9) for p in pool)


def fmt_numbers(nums: set[float]) -> str:
    return ", ".join(_fmt_number(n) for n in sorted(nums))


def _fmt_number(n: float) -> str:
    # '73 million' reads better in a reason than '7.3e+07'
    for scale, word in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if abs(n) >= scale:
            return f"{n / scale:g} {word}"
    return f"{n:g}"


def norm_entity(ent: str) -> str:
    return re.sub(r"^(the|a|an)\s+", "", fold(ent)).strip(" .,'\"")


def mentioned(ent: str, text: str) -> bool:
    """`ent` appears in the (folded) text as whole words: 'Ian Smith' is not in 'Brian Smithson'.

    A possessive or demonym ending still counts: 'Europe' is grounded by 'European sales'.
    """
    name = norm_entity(ent)
    return bool(name) and re.search(r"(?<!\w)" + re.escape(name) + r"(?:'s|s|n|an|ian|ese)?(?!\w)", text) is not None


def _split_title(name: str) -> tuple[str, str]:
    """'cfo desmond okafor' -> ('cfo', 'desmond okafor')."""
    m = _TITLES_RE.match(name)
    return (m.group(1).strip(), name[m.end():]) if m else ("", name)


def _canon_role(title: str) -> str:
    title = title.rstrip(".")
    return _ROLE_ABBR.get(title, title)


def _person_grounded(ent: str, text: str) -> bool:
    """A person's name counts without a leading title, unless the text gives that person a different role."""
    title, name = _split_title(norm_entity(ent))
    if not name or not mentioned(name, text):
        return False
    claim_roles = {_canon_role(r) for r in re.findall(rf"(?<![a-z])({_ROLE})\.?(?= |$)", title)}
    if not claim_roles:
        return True
    text_roles = set()
    for m in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text):
        before = _ROLE_BEFORE_RE.search(text[max(0, m.start() - 40):m.start()])
        if before:
            text_roles.add(_canon_role(before.group(1)))
    return not text_roles or bool(claim_roles & text_roles)


def _calendar_words(text: str) -> set[int]:
    return {_CALENDAR[w] for w in _CALENDAR_RE.findall(text)}


def grounded(ent: str, label: str, text: str, text_numbers: set[float]) -> bool:
    """The entity is stated in the folded text, or (for dates) its numbers and month names are."""
    if mentioned(ent, text) or (label == "PERSON" and _person_grounded(ent, text)):
        return True
    nums = numbers(ent)
    # 'August 1969' is not grounded by '1969' alone: the month must match too
    return bool(nums) and all(has_number(text_numbers, n) for n in nums) \
        and _calendar_words(fold(ent)) <= _calendar_words(text)


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
    source_lower, ev_lower = fold(source_lower), fold(ev_text)
    out = []

    mismatch = number_mismatch(claim_text, related[0] if quantities(related[0]) else ev_text)
    if mismatch:
        out.append(mismatch)

    claim_ent_text = " ".join(norm_entity(e) for e, _ in claim.entities)
    for ent, label in claim.entities:
        # an entity that appears anywhere in the sources is grounded, so it is not a conflict
        if grounded(ent, label, source_lower, set()) or grounded(ent, label, ev_lower, ev.numbers):
            continue
        rivals = [t for t, l in ev.entities if l == label and not mentioned(t, claim_ent_text)]
        if rivals:
            # prefer the rival that shares a word with the claim's entity ("Atlantic Ocean" -> "Pacific Ocean")
            words = set(norm_entity(ent).split())
            rival = max(rivals, key=lambda r: len(words & set(norm_entity(r).split())))
            out.append(f"{label} mismatch: claim says '{ent}', evidence says '{rival}'")

    if polarity_flip(nlp, claim_text, related[0]):
        out.append("polarity mismatch: negation differs from evidence")
    return out


_RAW_NUMBER_RE = re.compile(r"\d[\d,.]*\d|\d")


def same_values_other_notation(claim_text: str, evidence: str) -> bool:
    """Every claim number equals an evidence number, but at least one is written differently.

    '$2.1 billion' vs '$2,100 million' or '1,280.5' vs '1280.5'. The NLI model reads such pairs as a
    contradiction (p ~0.99), so the pipeline does not let NLI alone decide them.
    """
    claim_values = numbers(claim_text)
    if not claim_values or not all(has_number(numbers(evidence), v) for v in claim_values):
        return False
    claim_raw = set(_RAW_NUMBER_RE.findall(fold(claim_text)))
    evidence_raw = set(_RAW_NUMBER_RE.findall(fold(evidence)))
    words = set(re.findall(r"[a-z]+", fold(claim_text))) & set(WORD_NUMBERS)
    return bool(claim_raw - evidence_raw) or bool(words - set(re.findall(r"[a-z]+", fold(evidence))))


def number_mismatch(claim_text: str, evidence: str) -> str | None:
    """A claim number that differs from every evidence number counting the same thing.

    Numbers are compared with the single best-matching sentence: it states the same fact. A claim number
    with no same-kind number there ('14 percent' against a sentence about '$412 million', or a claim that
    joins two sentences) is no conflict; novel() still checks that it appears somewhere in the sources.
    """
    ev_q = quantities(evidence)
    ev_values = {q.value for q in ev_q}
    claim_vals, ev_vals = set(), set()
    for q in quantities(claim_text):
        if has_number(ev_values, q.value):
            continue
        # two bare numbers ('3 million' vs '2,000,000') are taken to count the same thing
        same_kind = {e.value for e in ev_q if (e.kinds & q.kinds) or not (e.kinds or q.kinds)}
        if same_kind:
            claim_vals.add(q.value)
            ev_vals |= same_kind
    if not claim_vals:
        return None
    return f"number mismatch: claim says {fmt_numbers(claim_vals)}, evidence says {fmt_numbers(ev_vals)}"


# ---------------------------------------------------------------- negation
_LIGHT_VERBS = {"be", "do", "have"}
_PREDICATE_DEPS = {"acomp", "attr", "oprd", "xcomp", "dobj"}     # what a copula / light verb actually asserts
_MAIN_DEPS = {"conj", "acomp", "attr", "oprd", "xcomp", "ccomp"}
_CONTENT_POS = {"NOUN", "PROPN", "NUM", "ADJ", "VERB", "ADV"}


def polarity_flip(nlp, claim: str, evidence: str) -> bool:
    """The main predicate the claim asserts is negated in the evidence, or the other way round.

    'was not compatible' vs 'was compatible' flips; a negation in a side clause ('which is not the tallest')
    or on a predicate the other text only uses in a side clause does not.
    """
    c, e = nlp(claim), nlp(evidence)
    return _flipped(e, c) or _flipped(c, e)


def _flipped(negated_doc, other_doc) -> bool:
    """A main-clause predicate negated in `negated_doc` is asserted, un-negated, in `other_doc`'s main clause."""
    other_negated = {t.lemma_.lower() for t in _negated_predicates(other_doc)}
    for pred in _negated_predicates(negated_doc):
        lemma = pred.lemma_.lower()
        if lemma in other_negated or not _is_main(pred):
            continue
        for tok in other_doc:
            if tok.lemma_.lower() == lemma and _is_main(tok):
                a, b = _frame(pred), _frame(tok)
                # the same predicate about the same arguments ('take two tablets' is not 'take it')
                if (a & b) or not (a or b):
                    return True
    return False


def _is_light(tok) -> bool:
    return tok.pos_ == "AUX" or tok.lemma_.lower() in _LIGHT_VERBS


def _negated_predicates(doc) -> list:
    """For 'X is not Y' the negation sits on the copula: the predicate is Y, not 'be'."""
    out = []
    for t in doc:
        if t.dep_ != "neg":
            continue
        head = t.head
        preds = [ch for ch in head.children if ch.dep_ in _PREDICATE_DEPS] if _is_light(head) else []
        out += preds or [head]
    return out


def _is_main(tok) -> bool:
    """The token is (part of) what the sentence asserts, not a relative, adverbial or subject clause."""
    while tok.dep_ != "ROOT":
        if tok.dep_ not in _MAIN_DEPS and not (tok.dep_ in _PREDICATE_DEPS and _is_light(tok.head)):
            return False
        tok = tok.head
    return True


def _frame(tok) -> set[str]:
    """Content words attached to a predicate (and to its copula): its subject, objects and modifiers."""
    heads = [tok] + ([tok.head] if tok.dep_ in _PREDICATE_DEPS and _is_light(tok.head) else [])
    words = set()
    for head in heads:
        for ch in head.children:
            if ch.i == tok.i:
                continue
            for w in [ch] + ([g for g in ch.children if g.dep_ == "pobj"] if ch.dep_ in ("prep", "agent") else []):
                if w.pos_ in _CONTENT_POS and w.dep_ != "neg":
                    words.add(w.lemma_.lower())
    return words


def novel(claim: Facts, source_lower: str) -> list[str]:
    """Numbers and names in the claim that appear in no source at all."""
    source_lower = fold(source_lower)
    src_nums = numbers(source_lower)
    out = [fmt_numbers({n}) for n in claim.numbers if not has_number(src_nums, n)]
    for ent, label in claim.entities:
        if not grounded(ent, label, source_lower, src_nums):
            out.append(f"'{ent}'")
    return out
