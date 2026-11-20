"""Claim decomposition: split a model response into atomic factual assertions.

A sentence is cut into several claims when it packs several facts:

* coordinated verbs     "Curie discovered polonium and won two Nobel Prizes."
                        -> "Curie discovered polonium." / "Curie won two Nobel Prizes."
* non-restrictive       "Curie, who was born in Warsaw, won the prize in 1903."
  relative clauses      -> "Curie won the prize in 1903." / "Curie was born in Warsaw."
* contrastive clauses   "The drug is cheap, although it causes headaches."
                        -> "The drug is cheap." / "It causes headaches."

Sentences with nothing to check (questions, pleasantries, headings) are dropped, openers such as
"In summary," and markdown list markers are ignored, and a subject pronoun is resolved to the
subject it refers to, so that "She died in 1934." is checked as "Marie Curie died in 1934.".
"""
from __future__ import annotations

import re
from typing import NamedTuple

from .schemas import Claim

# A pleasantry drops its whole sentence; an opener is only cut off, so the fact after it is still checked.
_PLEASANTRY = re.compile(      # "Here is the answer." is filler; "Here are the facts: ..." goes on to a fact
    r"^(i hope|hope this|let me know|feel free|great question|here(?: is|'s|’s| are)\b[^:]*$)", re.IGNORECASE)
_OPENER = re.compile(r"(in summary|in conclusion|overall|sure|certainly|of course)\s*[,!:]", re.IGNORECASE)
_LIST_ITEM = re.compile(r"^(\s*(?:[-*+•◦▪●‣⁃·–—]|\d+[.)])\s+)")
_HEADING = re.compile(r"^\s*#{1,6}\s")
# Bold, code, and single * or _ emphasis. A single marker counts only when it hugs a word,
# so "2 * 3", snake_case and a "* " list marker are left alone.
_EMPHASIS = re.compile(r"\*\*|__|`|(?<!\w)[*_](?=\S)|(?<=\S)[*_](?!\w)")
_LABEL = re.compile(r"^[A-Za-z][\w ()/&-]{0,40}:\s*\S")       # "Revenue: $412M"
_CLOSERS = "\"'”’»)]"                            # may follow the final ? or . of a sentence
_SENTENCE_END = (".", "!", "?", ";")
_NO_SPACE_BEFORE = (",", ".", ";", ":", "%", ")")
_SUBJECT = ("nsubj", "nsubjpass")
_CONTRAST = ("although", "though", "whereas")
_AUX = ("aux", "auxpass")
# Verbs whose bare -ed form is a past tense, not a passive participle, so a shared "was" does not fit them.
_INTRANSITIVE = frozenset({
    "die", "retire", "land", "survive", "improve", "expand", "arrive", "resign", "emigrate", "immigrate",
    "graduate", "decline", "grow", "remain", "return", "succeed", "fail", "live", "work", "serve", "marry",
    "move", "stay", "rise", "fall", "recover", "settle", "travel", "appear", "disappear", "occur", "happen",
    "exist", "collapse", "evolve", "flourish", "prosper", "perish", "vanish", "erupt", "retreat",
    "compete", "participate", "testify", "debut"})
# Heads that say little alone, so ", which ..." after "<head> of Y" describes the whole phrase, not Y.
_EMPTY_HEADS = frozenset({
    "one", "some", "most", "part", "all", "many", "few", "each", "none", "half", "rest", "majority", "minority",
    "price", "cost", "number", "share", "rate", "level", "value", "amount", "percentage", "proportion", "total",
    "city", "town", "state", "university", "republic", "kingdom", "province", "county", "island"})


def load_nlp(model: str = "en_core_web_sm"):
    import spacy

    try:
        return spacy.load(model)
    except OSError as exc:  # model not downloaded
        raise RuntimeError(f"spaCy model '{model}' is missing. Run: python -m spacy download {model}") from exc


class _Piece(NamedTuple):
    tokens: list            # reading order; a copied subject and shared auxiliary come first
    start: int              # character span of the piece's own words in the parsed block
    end: int
    subject: object         # subject token (or None), used to resolve a pronoun
    own_subject: bool       # the clause has its own subject, so a short piece is still a full claim
    words: dict             # token id -> text to show instead ("which" -> "this")
    context: tuple = ()     # tokens to check instead, when the piece cannot be checked alone ("This beat ...")


class ClaimExtractor:
    def __init__(self, nlp=None, min_words: int = 3):
        self.nlp = nlp or load_nlp()
        self.min_words = min_words

    def extract(self, response: str) -> list[Claim]:
        # Blank out inline markdown without moving any character, so spans still index the original text.
        clean = _EMPHASIS.sub(lambda m: " " * len(m.group()), response)
        claims: list[Claim] = []
        referents = _Referents()
        seen: dict[str, int] = {}
        for block_start, block_end in _blocks(clean):
            doc = self.nlp(clean[block_start:block_end])
            for sent in doc.sents:
                text = re.sub(r"\s+", " ", sent.text).strip()
                if not text or _is_question(text) or _PLEASANTRY.match(text):
                    continue
                referents.update(sent)      # first, so a pronoun later in this same sentence can use it
                pieces = _decompose(sent)
                for piece in pieces:
                    cut = _opener_length(piece.tokens)
                    tokens = piece.tokens[cut:]
                    body = _join(tokens, piece.words)
                    if not body or _PLEASANTRY.match(body) or not self._is_claim(body, piece, text, len(pieces) > 1):
                        continue
                    start = tokens[0].idx if cut else piece.start
                    claim = Claim(len(claims) + 1, _finish(body), text, block_start + start, block_start + piece.end)
                    referent = referents.resolve(piece.subject)
                    if referent and any(t.i == piece.subject.i for t in tokens):
                        claim.resolved_text = _finish(_join(tokens, {**piece.words, piece.subject.i: referent}))
                    if piece.context:       # also keeps two "This beat expectations." from different facts apart
                        claim.resolved_text = _finish(_join(piece.context[_opener_length(piece.context):]))
                    key = _dedupe_key(claim.resolved_text)
                    claim.duplicate_of = seen.get(key)
                    seen.setdefault(key, claim.id)
                    claims.append(claim)
        return claims

    def _is_claim(self, body: str, piece: _Piece, sentence: str, split: bool) -> bool:
        words = len(body.split())
        if words >= self.min_words:
            return True
        if _LABEL.match(body) and re.search(r"\d", body):     # "Revenue: $412M" is a fact, not a fragment
            return True
        # A clause split off with its own subject ("..., and Aldrin followed.") is a claim even when short.
        return split and piece.own_subject and words >= 2 and len(sentence.split()) >= self.min_words


# ------------------------------------------------------------------ text blocks
def _blocks(text: str) -> list[tuple[int, int]]:
    """Character ranges to parse separately: paragraphs, and each markdown list item on its own."""
    blocks: list[tuple[int, int]] = []
    para_start = para_end = None
    pos = 0
    for line in text.splitlines(keepends=True):
        start, end = pos, pos + len(line.rstrip("\r\n"))
        pos += len(line)
        item = _LIST_ITEM.match(line)
        if not line.strip() or _HEADING.match(line) or item:
            if para_start is not None:
                blocks.append((para_start, para_end))
                para_start = None
            if item:
                blocks.append((start + len(item.group(1)), end))
            continue
        if para_start is None:
            para_start = start
        para_end = end
    if para_start is not None:
        blocks.append((para_start, para_end))
    return blocks


# ------------------------------------------------------------------ sentence -> claims
def _decompose(sent) -> list[_Piece]:
    """Split one sentence into pieces: main clause, coordinated verbs, contrastive and relative clauses."""
    root = sent.root
    excluded: set[int] = set()
    relative: list[_Piece] = []
    for pronoun in sent:
        found = None if pronoun.i in excluded else _relative_clause(sent, pronoun)
        if found:
            relative.append(found[0])
            excluded |= found[1]
    contrast: list[_Piece] = []
    for verb in sent:
        found = _contrast_clause(sent, verb, excluded)
        if found:
            contrast.append(found[0])
            excluded |= found[1]

    conj_verbs = _verb_conjuncts(root, excluded)
    if not conj_verbs and not relative and not contrast:
        tokens = [t for t in sent if not t.is_space]
        return [_piece(tokens, tokens, _subject(root), True)] if tokens else []

    main_excluded = set(excluded)
    for verb in conj_verbs:                     # the whole chain hanging off each conjunct leaves the main clause
        main_excluded |= {t.i for t in verb.subtree} | _leading(root, verb) | _separators(root, verb)
    pieces: list[_Piece] = []
    main = _trim([t for t in sent if t.i not in main_excluded and _keep(t)])
    if main:
        pieces.append(_piece(main, main, _subject(root), _subject(root) is not None))
    for verb in _all_conjuncts(root, excluded):
        ids = _clause_ids(verb.head, verb)
        own = _trim([t for t in sent if t.i in ids and t.i not in excluded and _keep(t)])
        if not own:
            continue
        subject = _subject(verb)
        if subject is not None or _has_own_subject(verb):
            pieces.append(_piece(own, own, subject, True))
            continue
        subject = _inherited_subject(verb)
        copied = _trim([t for t in _subtree(subject) if t.i not in excluded and _keep(t)]) if subject is not None else []
        pieces.append(_piece(copied + _shared_aux(verb) + own, own, subject, False))
    return pieces + contrast + relative


def _piece(tokens: list, own: list, subject, own_subject: bool, words: dict[int, str] | None = None,
           context: list | None = None) -> _Piece:
    return _Piece(tokens, own[0].idx, own[-1].idx + len(own[-1]), subject, own_subject, words or {},
                  tuple(context or ()))


def _subtree(token) -> list:
    # spaCy yields a non-projective subtree out of order ("which X credited to lower costs"): sort it.
    return sorted(token.subtree, key=lambda t: t.i)


def _keep(t) -> bool:
    return not t.is_space and t.text not in _SENTENCE_END


def _trim(tokens: list) -> list:
    """Drop commas and semicolons left dangling at either end after a clause was cut out."""
    while tokens and tokens[0].text in (",", ";"):
        tokens = tokens[1:]
    while tokens and tokens[-1].text in (",", ";"):
        tokens = tokens[:-1]
    return tokens


def _subject(verb):
    return next((c for c in verb.children if c.dep_ in _SUBJECT), None)


def _has_own_subject(verb) -> bool:
    return any(c.dep_ in ("nsubj", "nsubjpass", "expl", "csubj") for c in verb.children)


def _inherited_subject(verb):
    """The subject a conjunct shares: "Einstein developed X, won Y, and moved to Z" -> Einstein for all three."""
    while verb.dep_ == "conj" and _subject(verb) is None and verb.head.i != verb.i:
        verb = verb.head
    return _subject(verb)


# --- coordinated verbs
def _verb_conjuncts(head, excluded: set[int]) -> list:
    return [t for t in head.children if t.dep_ == "conj" and t.pos_ in ("VERB", "AUX") and t.i not in excluded
            and _split_keeps_polarity(t)]


def _split_keeps_polarity(verb) -> bool:
    """A conjunct under a shared "not" / "never" is split off only when the negation can go with it.

    "did not raise prices or cut jobs" means neither happened, so each piece keeps "did not". With "and"
    (or "but", or bare commas) the scope is unclear: "did not raise prices and cut jobs" may say the jobs
    were cut, so the coordination stays one claim rather than risk reversing a piece."""
    if not _under_negation(verb):
        return True
    return _coordinator(verb) in ("or", "nor") and any(t.dep_ == "neg" for t in _shared_aux(verb))


def _shares_head(verb) -> bool:
    """A conjunct with no subject, auxiliary or negation of its own takes them from the verb before it."""
    return verb.dep_ == "conj" and not _has_own_subject(verb) and not any(
        c.dep_ in _AUX + ("neg",) for c in verb.children)


def _under_negation(verb) -> bool:
    if not _shares_head(verb):
        return False
    head = verb.head
    prefix = _prefix(head)
    neither = any(c.dep_ == "preconj" and c.lower_ == "neither" for c in head.children)
    if neither or any(t.dep_ == "neg" for t in prefix):
        return True
    return not prefix and _under_negation(head)      # "did not A, B or C": C sits under B, which sits under A


def _coordinator(verb) -> str | None:
    """The word joining a conjunct to the verb before it; in "A, B or C" the "or" before C also joins B."""
    head = verb.head
    before = [c for c in head.children if c.dep_ == "cc" and head.i < c.i < verb.i]
    after = [c for c in verb.children if c.dep_ == "cc" and c.i > verb.i]
    found = before or after
    return found[-1].lower_ if found else None


def _all_conjuncts(root, excluded: set[int]) -> list:
    """Every verb in the coordination chain. spaCy hangs a third verb off the second, not off the root."""
    found, frontier = [], [root]
    while frontier:
        for verb in _verb_conjuncts(frontier.pop(0), excluded):
            found.append(verb)
            frontier.append(verb)
    return found


def _separators(head, verb) -> set[int]:
    """The "and" / commas of head right before its conjunct (not commas that close an earlier phrase)."""
    seps = [t for t in head.children if t.dep_ in ("cc", "punct") and head.i < t.i < verb.i]
    ccs = [t.i for t in seps if t.dep_ == "cc"]
    end = max(ccs) if ccs else verb.i           # words after the "and" go with the conjunct (see _leading)
    words = [c.right_edge.i for c in head.children if c.dep_ not in ("cc", "punct") and c.i < end]
    boundary = max([head.i] + words)
    return {t.i for t in seps if t.i > boundary}


def _leading(head, verb) -> set[int]:
    """Words of head placed after the "and" but before the conjunct verb belong to the conjunct:
    "..., and in clinical studies 3 in 100 patients stopped" (spaCy attaches "in" to the first verb)."""
    seps = _separators(head, verb)
    if not seps:
        return set()
    boundary = max(seps)
    return {t.i for c in head.children if boundary < c.i < verb.i and c.dep_ not in ("cc", "punct")
            for t in c.subtree}


def _clause_ids(head, verb) -> set[int]:
    """Token ids of one conjunct clause, without the conjuncts nested under it."""
    ids = {t.i for t in verb.subtree} | _leading(head, verb)
    for nested in _verb_conjuncts(verb, set()):
        ids -= {t.i for t in nested.subtree} | _leading(verb, nested) | _separators(verb, nested)
    return ids


def _shared_aux(verb) -> list:
    """Auxiliaries and negation a conjunct shares with the verb before it: "claims are repaired or [are]
    replaced", "did not raise prices or [did not] cut jobs"."""
    if not _shares_head(verb):
        return []
    head = verb.head
    shared = _prefix(head) or _shared_aux(head)
    # An auxiliary only fits a verb of the same form; "never" alone fits any finite verb.
    if any(t.dep_ in _AUX for t in shared) and _verb_form(verb) != _verb_form(head):
        return []
    return shared


def _prefix(head) -> list:
    """The auxiliaries and negation in front of a verb, with a contraction kept whole ("ca" + "n't")."""
    found = {c.i: c for c in head.children if c.dep_ in _AUX + ("neg",) and c.i < head.i}
    doc = head.doc
    for aux in [t for t in found.values() if t.dep_ in _AUX]:
        if aux.i + 1 < len(doc) and doc[aux.i + 1].dep_ == "neg":     # never copy a bare "ca" / "wo"
            found[aux.i + 1] = doc[aux.i + 1]
    return [found[i] for i in sorted(found)]


def _verb_form(verb) -> str:
    # A regular past form with no object is often a participle the tagger called VBD ("was recruited and tested"),
    # unless the verb is one that rarely takes an object: "was born ... and died" must not become "was died".
    if (verb.tag_ == "VBD" and verb.lower_.endswith("ed") and verb.lemma_.lower() not in _INTRANSITIVE
            and not any(c.dep_ == "dobj" for c in verb.children)):
        return "VBN"
    return verb.tag_


# --- relative and contrastive clauses
def _relative_clause(sent, pronoun) -> tuple[_Piece, set[int]] | None:
    """A non-restrictive relative clause: ", who was born in Warsaw," or ", which the CFO credited to costs,"."""
    doc = sent.doc
    if pronoun.tag_ not in ("WDT", "WP") or pronoun.lower_ not in ("who", "whom", "which", "that"):
        return None
    if pronoun.i - 2 < sent.start or doc[pronoun.i - 1].text != ",":
        return None                         # restrictive ("the man who ...") defines the noun: keep it whole
    is_subject = pronoun.dep_ in _SUBJECT
    if not is_subject and (pronoun.dep_ != "dobj" or pronoun.lower_ == "that"):
        return None
    gov = verb = pronoun.head               # gov is the verb "which" belongs to; verb heads the relative clause
    while verb.dep_ in ("xcomp", "ccomp") and verb.head.i > pronoun.i:
        verb = verb.head                    # spaCy may hang an object "which" off an inner verb
    if verb.dep_ not in ("relcl", "advcl") or verb.pos_ not in ("VERB", "AUX") or verb.left_edge.i != pronoun.i:
        return None
    if is_subject and verb.i != gov.i:
        return None                         # ", which experts believe is safe," has no subject slot to fill: keep whole
    clause = [t for t in _subtree(verb) if _keep(t)]
    clause_ids = {t.i for t in verb.subtree}
    body = [t for t in clause if t.i != pronoun.i]
    if not body:
        return None
    antecedent = _antecedent(doc[pronoun.i - 2], clause_ids, pronoun, verb if is_subject else None)
    if antecedent:
        stand_in, context = antecedent, []
        first = antecedent[0]               # "The telescope, which NASA launched" -> "NASA launched the telescope"
        words = {first.i: first.lower_} if first.i == sent.start and first.pos_ == "DET" else {}
    else:
        # With no noun phrase right before the comma, "which" stands for the whole clause. "This beat expectations"
        # says nothing checkable alone, so it is checked together with the clause it describes.
        stand_in, words = [pronoun], {pronoun.i: "this"}
        context = [t for t in sent if t.i <= max(clause_ids) and _keep(t)]
    if is_subject:
        tokens = stand_in + body
    else:                                   # the object goes where it was taken from: "signed off on the merger"
        at = sum(1 for t in body if t.i <= _object_slot(gov))
        tokens = body[:at] + stand_in + body[at:]
    right = verb.right_edge
    excluded = clause_ids | {pronoun.i - 1}
    if right.i + 1 < sent.end and doc[right.i + 1].text == ",":
        excluded.add(right.i + 1)
    return _piece(tokens, clause, None, True, words, context), excluded


def _object_slot(gov) -> int:
    """The token the moved object follows: the verb, its particle, or a stranded preposition ("signed off on")."""
    prt = [c.i for c in gov.children if c.dep_ == "prt"]
    stranded = [c.i for c in gov.children if c.dep_ == "prep" and c.i > gov.i and not list(c.children)]
    return max([gov.i] + prt + stranded)


def _antecedent(anchor, clause_ids: set[int], pronoun, verb=None) -> list | None:
    """The noun phrase right before ", which ...". That is what the clause describes, even when spaCy
    attaches the clause higher up ("a better choice than ibuprofen, which ..." describes ibuprofen)."""
    head = anchor
    while head.dep_ in ("compound", "nummod", "flat") and head.head.pos_ in ("NOUN", "PROPN"):
        head = head.head                                    # "Python 3.0" -> Python
    if head.pos_ not in ("NOUN", "PROPN", "PRON"):         # "in 2005, which ...", "a year ago, which ..."
        return None
    while _describes_outer_noun(head, pronoun, verb):
        head = head.head.head                               # "the CEO of Acme, who ..." -> CEO
    return [t for t in _subtree(head) if t.i <= anchor.i and t.i not in clause_ids
            and t.dep_ not in ("mark", "cc") and t.text not in (",", ";")]


def _describes_outer_noun(noun, pronoun, verb) -> bool:
    """Whether ", which / who ..." after "X of Y" (noun is Y) describes X. By default it describes Y:
    "the capital of France, which has 68 million people" is about France."""
    if noun.dep_ != "pobj" or noun.head.lower_ != "of" or noun.head.head.pos_ not in ("NOUN", "PROPN", "PRON"):
        return False
    outer = noun.head.head
    number = _verb_number(verb)
    if number and _noun_number(outer) not in (None, number) and _noun_number(noun) == number:
        return False                                        # "the works of Picasso, who was born ..." is singular
    if outer.lemma_.lower() in _EMPTY_HEADS or _same_entity(noun, outer):
        return True                                         # "the price of oil", "the University of Oxford"
    if pronoun.lower_ in ("who", "whom"):                   # "the CEO of Acme, who" is the CEO, not Acme
        return not (noun.ent_type_ == "PERSON" and outer.ent_type_ != "PERSON")
    return False


def _verb_number(verb) -> str | None:
    """"sing" / "plur" when the relative clause's finite verb shows it ("was", "has", "were")."""
    if verb is None:
        return None
    for t in [c for c in verb.children if c.dep_ in _AUX] + [verb]:
        if t.tag_ == "VBZ" or t.lower_ == "was":
            return "sing"
        if t.tag_ == "VBP" or t.lower_ == "were":
            return "plur"
    return None


def _noun_number(noun) -> str | None:
    return {"NN": "sing", "NNP": "sing", "NNS": "plur", "NNPS": "plur"}.get(noun.tag_)


def _same_entity(a, b) -> bool:
    return any(e.start <= a.i < e.end and e.start <= b.i < e.end for e in a.doc.ents)


def _contrast_clause(sent, verb, excluded: set[int]) -> tuple[_Piece, set[int]] | None:
    """", although Y" with its own subject is a separate fact: a true Y must not hide a false X."""
    if verb.dep_ != "advcl" or verb.head.i != sent.root.i or _subject(verb) is None:
        return None
    marks = [c for c in verb.children if c.dep_ == "mark" and c.lower_ in _CONTRAST]
    if not marks:
        return None
    clause = _subtree(verb)
    drop = {m.i for m in marks} | {c.i for c in verb.children if c.dep_ == "advmod" and c.lower_ == "even"}
    body = _trim([t for t in clause if t.i not in drop and t.i not in excluded and _keep(t)])
    if not body:
        return None
    doc, left, right = sent.doc, clause[0], clause[-1]
    ids = {t.i for t in clause}
    if left.i - 1 >= sent.start and doc[left.i - 1].text == ",":
        ids.add(left.i - 1)
    if right.i + 1 < sent.end and doc[right.i + 1].text == ",":
        ids.add(right.i + 1)
    return _piece(body, body, _subject(verb), True), ids


# ------------------------------------------------------------------ pronouns
class _Referents:
    """Remembers recent sentence subjects so a subject pronoun can be replaced by what it refers to."""

    def __init__(self):
        self.person = self.thing = self.plural = None
        self.rivals: dict = {}      # kind -> latest named object that the same pronoun could also mean

    @staticmethod
    def _subject(sent):
        return _subject(sent.root)

    def resolve(self, subj) -> str | None:
        """The referent for a he / she / it / they subject token, when one is known and not ambiguous."""
        if subj is None or subj.pos_ != "PRON" or _is_dummy_it(subj):
            return None
        kind = _PRONOUN_KIND.get(subj.lower_)
        if kind is None or self._ambiguous(kind, subj):
            return None
        return getattr(self, kind)

    def _ambiguous(self, kind: str, subj) -> bool:
        # "The FDA approved Zestril, and it became ...": "it" may be Zestril, so guessing the FDA could invent
        # a contradiction. A rival later in the pronoun's own clause ("She married Pierre") does not count.
        rival = self.rivals.get(kind)
        return rival is not None and (rival.doc is not subj.doc or rival.i < subj.i)

    def update(self, sent) -> None:
        subj = self._subject(sent)
        if subj is not None and subj.pos_ != "PRON":
            ent = next((e for e in sent.ents if e.start <= subj.i < e.end), None)
            chunk = next((c for c in sent.noun_chunks if c.start <= subj.i < c.end), None)
            phrase = ent.text if ent is not None else (chunk.text if chunk is not None else subj.text)
            if ent is not None and ent.label_ == "PERSON":
                kind = "person"
            elif ent is None and subj.tag_ in ("NNS", "NNPS"):  # "the astronauts"; a company name is an "it"
                kind = "plural"
            else:
                kind = "thing"
            setattr(self, kind, phrase)
            self.rivals.pop(kind, None)
        for obj in sent:
            for kind in _rival_kinds(obj):
                self.rivals[kind] = obj


_PRONOUN_KIND = {"he": "person", "she": "person", "it": "thing", "they": "plural"}
_NUMERIC_ENTS = ("DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL")


def _rival_kinds(token) -> set[str]:
    """Pronoun kinds a named object could be referred to by: "approved Zestril" -> it, "married Pierre Curie" -> he.
    Only objects of a verb count; "born in Warsaw" does not compete with the subject."""
    is_object = token.dep_ in ("dobj", "dative") or (token.dep_ == "attr" and token.head.lemma_ != "be")
    named = token.pos_ == "PROPN" or (token.ent_type_ != "" and token.ent_type_ not in _NUMERIC_ENTS)
    if not is_object or not named:
        return set()
    if token.ent_type_ == "PERSON":
        return {"person"}
    return {"thing", "plural"} if token.tag_ in ("NNS", "NNPS") else {"thing"}   # "acquired Beats": it or they


def _is_dummy_it(subj) -> bool:
    """'It' that stands for a clause, not a thing: "It is estimated / true / important to note that ..."."""
    if subj.lower_ != "it":
        return False
    if subj.dep_ == "expl":
        return True
    verb = subj.head
    deps = {c.dep_ for c in verb.children}
    if subj.dep_ == "nsubjpass" and "ccomp" in deps:
        return True
    if verb.lemma_ == "be" and deps & {"acomp", "attr"}:
        # "It is easy to learn" still means the thing; "It is important to note that ..." does not.
        xcomp_clause = any(c.dep_ == "xcomp" and any(g.dep_ == "ccomp" for g in c.children) for c in verb.children)
        if deps & {"ccomp", "csubj"} or xcomp_clause:
            return True
    return verb.lemma_ in ("seem", "appear", "happen") and "ccomp" in deps


# ------------------------------------------------------------------ text helpers
def _join(tokens: list, replace: dict[int, str] | None = None) -> str:
    """Rebuild text from tokens: original spacing between neighbours, one space where words were cut out."""
    replace = replace or {}
    out: list[str] = []
    prev = None
    for t in tokens:
        if prev is not None:
            if t.i == prev.i + 1:
                out.append(prev.whitespace_)
            elif t.text not in _NO_SPACE_BEFORE:
                out.append(" ")
        out.append(replace.get(t.i, t.text))
        prev = t
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _opener_length(tokens: list) -> int:
    """How many leading tokens form an opener such as "In summary," (0 if none)."""
    for n in range(1, min(4, len(tokens)) + 1):
        if _OPENER.fullmatch(_join(tokens[:n])):
            return n
    return 0


def _is_question(text: str) -> bool:
    return text.rstrip(_CLOSERS).endswith("?")


def _dedupe_key(text: str) -> str:
    # Punctuation is ignored except right before a digit, so "1.5" / "15" and "-89" / "89" stay different.
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s](?!\d)", "", text.lower())).strip()


def _finish(text: str) -> str:
    text = text[0].upper() + text[1:]
    return text if text.rstrip(_CLOSERS).endswith((".", "!")) else text + "."
