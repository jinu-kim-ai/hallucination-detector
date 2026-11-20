"""Checkable facts inside a claim: numbers, named entities and negation.

Embeddings say whether a claim and a sentence are *about the same thing*. They do not notice
that "45 kilograms" is not "21.5 kilograms". These checks do.
"""
from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass, field

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
# Ordinals name a rank or position ('the first person', 'First, ...', 'the second largest'), not an amount.
_ORDINAL_WORDS = {"first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
                  "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
                  "eighteenth", "nineteenth", "twentieth"}
# Kinds that say exactly what a number is; such a number never stands in for an unlabelled one.
_STRONG_KINDS = {"money", "%", "time", "calendar year", "day"}
# Nouns that all count people: '300 workers' and '1,500 people' state the same fact. 'member' is left out:
# '5,000 members' of a club and '27 member states' are not a head count of the same people.
_PERSON_NOUNS = {"people", "person", "employee", "worker", "staff", "staffer", "personnel", "individual"}
# A preposition (or a month or weekday name, see _kinds) before a 4-digit number makes it a year even when a
# noun follows ('in 1998 workers ...', 'In March 2020 workers ...'). An article does not: 'the 1500 employees'
# is a head count as often as 'the 2019 workers' strike' is a year, so such a number keeps both kinds.
_YEAR_PREPS = {"in", "by", "since", "from", "until", "till", "during", "of", "before", "after", "circa"}
# Words just before a number (or just after it) that phrase it as a fall: 'fell 3.2%' is the source's '-3.2%'.
_DECREASE_BEFORE_RE = re.compile(
    r"\b(?:fell|falls?|falling|declined|declines?|declining|decreased|decreases?|decreasing|dropped|drops?|"
    r"dropping|down|lost|loses?|losing|shrank|shrunk|shrinks?|shrinking|slipped|slips?|slipping|contracted|"
    r"contracts?|contracting|slid|slides?|sliding|an? (?:decline|decrease|drop|fall|loss|reduction|contraction) of)"
    r"(?: by)? $")
_DECREASE_AFTER_RE = re.compile(r"(?: ?%| percent| per cent)?[ -](?:decline|decrease|drop|fall|loss|reduction|"
                                r"contraction)\b")
# A numeral with an optional sign, currency, decimal part and scale: '-40', '$1,234.56', '$1.6 billion', '1.2M'.
# The sign only counts when nothing word-like is glued to its left, so '1990-1995' and 'COVID-19' stay positive.
# Lower-case k/m/b only scale after a currency sign, so '5 m' (meters) is not 5 million.
_NUM_RE = re.compile(
    r"(?<![\w.])(?P<sign>-)?(?P<cur>[$\u20ac\u00a3])?(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    # '$5 million' and '$5-million'; not 'the 2021 billion-dollar deal', where the scale belongs to a compound
    r"(?:[\s-]?(?P<scale>(?i:thousand|million|billion|trillion))\b(?!-)"
    r"|(?P<abbr>(?(cur)(?i:bn|k|m|b)|(?:bn|K|M|B)))\b"
    r"|(?P<ord>st|nd|rd|th)\b)?")
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
# Month names that are also ordinary words ('which may cause', 'protesters march'): in a source they only
# count as a month next to a number or after a word that introduces a date.
_AMBIGUOUS_MONTHS = {"may", "march", "mar"}
_MONTH_NAMES = "|".join(sorted({k for k, v in _CALENDAR.items() if v < 12}, key=len, reverse=True))
_PLAIN_MONTH_NAMES = "|".join(sorted({k for k, v in _CALENDAR.items() if v < 12} - _AMBIGUOUS_MONTHS,
                                     key=len, reverse=True))
# A day number right before a month: '4 November', 'the 5th of May', '5 May 2019' (not '65 may experience').
_DAY_BEFORE_MONTH_RE = re.compile(rf",? (?:of (?:{_MONTH_NAMES})|(?:{_PLAIN_MONTH_NAMES})|(?:may|march|mar)(?=,? \d))\b")
_MONTH_BEFORE_RE = re.compile(r"(?:\d(?:st|nd|rd|th)?,? (?:of )?|\b(?:in|early|late|mid|of|since|until|during|by)[ -])$")
_MONTH_AFTER_RE = re.compile(r"\.?,? (?:the )?\d")
# ...or as the start of a range opened by 'from' / 'between' AND closed by a second month name: 'from march to
# june', 'between may and july'. A bare 'and may' or 'march through' stays a verb: 'dipped in april and may
# recover', 'plan to march through june', 'the choice between may and must'.
_RANGE_START_BEFORE_RE = re.compile(r"\b(from|between) $")
_RANGE_START_AFTER_RE = {"from": re.compile(rf" (?:to|through|until) (?:{_MONTH_NAMES})\b"),
                         "between": re.compile(rf" and (?:{_MONTH_NAMES})\b")}
# Dates written with numbers only: '2019-11-04', '2019/11/04', '03/15/2022' (either day/month order).
_ISO_DATE_RE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
_SLASH_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
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
    numbers: set[float]                 # amounts, dates and labels the sources must contain (ordinals left out)
    entities: list[tuple[str, str]]     # (text, spaCy label)
    decreasing: set[float] = field(default_factory=set)     # numbers phrased as a fall: 'fell 3.2%' -> 3.2


@dataclass(frozen=True)
class Quantity:
    value: float
    kinds: frozenset[str]     # what the number counts: '%', 'money', 'calendar year', 'kilogram', ... (empty if unknown)
    ordinal: bool = False     # a rank or position ('third', '3rd'): compared only with other ordinals
    decrease: bool = False    # phrased as a fall ('fell 3.2%'), so it also stands for -3.2


def fold(text: str) -> str:
    """Lower-case text with typographic quotes and dashes made plain, for substring checks."""
    return re.sub(r"\s+", " ", text.translate(_TRANS)).lower()


def numbers(text: str) -> set[float]:
    """Numeric values in the text: '1,200' -> 1200, '12%' -> 12, '$1.6 billion' -> 1.6e9, 'twenty-five' -> 25."""
    return {q.value for q in quantities(text)}


def plain_numbers(text: str) -> set[float]:
    """numbers() without ordinals: 'First, ...' or 'the second largest' states no amount to verify."""
    return {q.value for q in quantities(text) if not q.ordinal}


def quantities(text: str) -> list[Quantity]:
    """Every number in the text with what it counts."""
    return [q for _, _, q in _scan(text)[1]]


def _scan(text: str) -> tuple[str, list[tuple[int, int, Quantity]]]:
    """The whitespace-normalized text and every number in it as (start, end, quantity)."""
    t = re.sub(r"\s+", " ", text.translate(_TRANS))
    low = t.lower()
    out, masked = [], list(low)
    for m in _NUM_RE.finditer(t):
        out.append((m.start(), m.end(), _numeral(m, t, low)))
        masked[m.start():m.end()] = " " * (m.end() - m.start())     # 'million' in '$2 million' is not read again
    out += _word_quantities(t, low, "".join(masked))
    return t, out


def _numeral(m: re.Match, t: str, low: str) -> Quantity:
    value = float(m["num"].replace(",", ""))
    scale = m["scale"] or m["abbr"]
    if scale:
        value *= _SCALES[scale.lower()]
    if m["sign"]:
        value = -value
    is_year = re.fullmatch(r"\d{4}", m["num"]) is not None and 1000 <= value <= 2100 and not (scale or m["sign"])
    kinds = _kinds(t, low, m.start(), m.end(), bool(m["cur"]), is_year)
    # 'July 20th' is a day of the month, not a rank
    return Quantity(value, kinds, bool(m["ord"]) and "day" not in kinds, _is_decrease(low, m.start(), m.end()))


def _word_quantities(t: str, low: str, masked: str) -> list[tuple[int, int, Quantity]]:
    """Number words: 'two' -> 2, 'seventy-five' -> 75, 'third' -> 3, 'one hundred and twenty' -> 120."""
    words = list(re.finditer(r"[a-z]+", masked))
    out, i = [], 0
    while i < len(words):
        value, j = _word_value(words, i, masked)
        if value is None:
            i += 1
            continue
        value, j = _compound(words, j, masked, value)
        start, end = words[i].start(), words[j - 1].end()
        kinds = _kinds(t, low, start, end, False, False)
        ordinal = words[j - 1].group() in _ORDINAL_WORDS and "day" not in kinds
        out.append((start, end, Quantity(value, kinds, ordinal, _is_decrease(low, start, end))))
        i = j
    return out


def _joined(words: list[re.Match], j: int, masked: str) -> bool:
    """Word j exists and follows word j-1 after a single space."""
    return 0 < j < len(words) and masked[words[j - 1].end():words[j].start()] == " "


def _scale_at(words: list[re.Match], j: int, masked: str) -> float:
    """The value of a scale word ('hundred', 'million') at j that joins the words before it, else 0."""
    if not _joined(words, j, masked) or words[j].group() not in WORD_NUMBERS:
        return 0
    return _SCALES.get(words[j].group(), 0)


def _compound(words: list[re.Match], j: int, masked: str, current: float) -> tuple[float, int]:
    """Extend the number word before j with scales and the groups between them.

    'one hundred and twenty' -> 120, 'one thousand nine hundred' -> 1900, 'two million three hundred
    thousand' -> 2.3e6. 'between one hundred and two hundred' stays two numbers.
    """
    total = 0.0
    while scale := _scale_at(words, j, masked):
        if not 0 < current < scale:
            break
        if scale == 100:
            current *= 100
        else:
            total, current = total + current * scale, 0.0
        j += 1
        group, j = _group_after_scale(words, j, masked, scale)
        current += group
    return total + current, j


def _group_after_scale(words: list[re.Match], j: int, masked: str, scale: float) -> tuple[float, int]:
    """The tens/units group after a scale word: 'hundred [and] twenty', 'thousand nine (hundred)'."""
    k = j + 1 if _joined(words, j, masked) and words[j].group() == "and" else j
    if not _joined(words, k, masked) or _scale_at(words, k, masked):
        return 0.0, j
    value, end = _word_value(words, k, masked)
    if value is None or value >= 100 or value != int(value):
        return 0.0, j
    next_scale = _scale_at(words, end, masked)
    # after 'and': 'and twenty' joins, 'and two hundred' starts a new number, and 'hundred and fifty
    # thousand' joins too (250,000): _compound then applies the thousand to all of 'two hundred and fifty'
    if k > j:
        take = next_scale < 100 or (scale == 100 and next_scale > 100)
    else:           # 'hundred twenty'; after a larger scale only a group with its own smaller scale
        take = scale == 100 or 0 < next_scale < scale
    return (value, end) if take else (0.0, j)


def _is_decrease(low: str, start: int, end: int) -> bool:
    return bool(_DECREASE_BEFORE_RE.search(low[max(0, start - 30):start]) or _DECREASE_AFTER_RE.match(low, end))


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


_MONEY_AFTER_RE = re.compile(r" ?(dollars?|usd|euros?|pounds?)\b")
_PERCENT_AFTER_RE = re.compile(r" ?(%|percent\b|per cent\b)")
_TIME_AFTER_RE = re.compile(r":\d")
_TIME_BEFORE_RE = re.compile(r"\d:$")
_NOUN_AFTER_RE = re.compile(r"[ -]+([a-z]+)")
_WORD_BEFORE_RE = re.compile(r"([A-Za-z]+)[ -]*$")
# How far before a number _kinds looks for the word that labels it. The patterns only need the nearest
# word; reading the whole prefix made every number cost the length of the text (quadratic on long sources).
_WINDOW = 80


def _kinds(t: str, low: str, start: int, end: int, money: bool, is_year: bool) -> frozenset[str]:
    """What a number counts, from the text around it (patterns after it are matched in place, without
    copying the rest of the text; the word before it is looked for only within _WINDOW characters)."""
    if money or _MONEY_AFTER_RE.match(low, end):
        return frozenset({"money"})
    if _PERCENT_AFTER_RE.match(low, end):
        return frozenset({"%"})
    if _TIME_AFTER_RE.match(low, end) or _TIME_BEFORE_RE.search(low, max(0, start - 2), start):
        return frozenset({"time"})
    if _DAY_BEFORE_MONTH_RE.match(low, end):
        return frozenset({"day"})                               # '4 November', 'the 5th of May'
    nxt = _NOUN_AFTER_RE.match(low, end)
    noun = _singular(nxt.group(1)) if nxt and nxt.group(1) not in _STOP else None
    prev = _WORD_BEFORE_RE.search(t, max(0, start - _WINDOW), start)
    if is_year:
        # not the "year" of "12-year-old". '1800 workers' may be a count, but 'in 1998 workers' and
        # 'March 2020 workers' are years.
        pw = prev.group(1).lower() if prev else ""
        dated = not noun or pw in _YEAR_PREPS or pw in _CALENDAR
        return frozenset({"calendar year"} if dated else {"calendar year", noun})
    kinds = {noun} if noun else set()
    if prev:
        word = prev.group(1).lower()
        if word in _CALENDAR and _CALENDAR[word] < 12:
            kinds.add("day")                                    # 'July 20'
        elif word in _LABEL_NOUNS or (prev.group(1)[0].isupper() and word not in _STOP and not word.endswith("ly")):
            kinds.add(word)                                     # 'Type 1', 'Apollo 11', 'Python 3.0'
    return frozenset(kinds)


def _singular(word: str) -> str:
    """The noun's singular, with every word for counted people as 'person'."""
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        word = word[:-1]
    return "person" if word in _PERSON_NOUNS else word


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

    A possessive, plural or demonym ending still counts: 'Europe' is grounded by 'European sales' and by
    'most Europeans'.
    """
    name = norm_entity(ent)
    return bool(name) and re.search(r"(?<!\w)" + re.escape(name) + r"(?:'s|s|(?:n|an|ian|ese)s?)?(?!\w)",
                                    text) is not None


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


def _source_calendar_words(text: str) -> set[int]:
    """Month and weekday names in a source. 'may' or 'march' counts only where it reads as a date."""
    out = set()
    for m in _CALENDAR_RE.finditer(text):
        word, before = m.group(1), text[max(0, m.start() - 12):m.start()]
        opener = _RANGE_START_BEFORE_RE.search(before)
        if word not in _AMBIGUOUS_MONTHS or _MONTH_BEFORE_RE.search(before) or _MONTH_AFTER_RE.match(text, m.end()) \
                or (opener and _RANGE_START_AFTER_RE[opener.group(1)].match(text, m.end())):
            out.add(_CALENDAR[word])
    return out


def _numeric_dates(text: str) -> set[tuple[int, int, int]]:
    """(year, month index, day) of every all-number date: '2019-11-04', and both readings of '03/04/2022'."""
    found = {(int(y), int(m) - 1, int(d)) for y, m, d in _ISO_DATE_RE.findall(text)}
    for a, b, y in _SLASH_DATE_RE.findall(text):
        found |= {(int(y), int(a) - 1, int(b)), (int(y), int(b) - 1, int(a))}
    return {(y, m, d) for y, m, d in found if 0 <= m < 12 and 1 <= d <= 31}


def _in_numeric_date(nums: set[float], months: set[int], text: str) -> bool:
    """'November 4, 2019' is the source's '2019-11-04', but not its '2019-04-11'."""
    months = {m for m in months if m < 12}
    years = {int(n) for n in nums if n.is_integer() and 1000 <= n <= 2100}
    days = {int(n) for n in nums if n.is_integer() and 1 <= n <= 31}
    if len(months) != 1 or len(years) > 1 or len(days) > 1 or len(years) + len(days) != len(nums):
        return False
    return any(m in months and (not years or y in years) and (not days or d in days)
               for y, m, d in _numeric_dates(text))


def grounded(ent: str, label: str, text: str, text_numbers: set[float]) -> bool:
    """The entity is stated in the folded text, or (for dates) its numbers and month names are."""
    if mentioned(ent, text) or (label == "PERSON" and _person_grounded(ent, text)):
        return True
    # 'The Nineteenth Amendment' is the source's '19th Amendment': with every number in digits on both
    # sides, the whole name (its other words included) must still appear
    if quantities(ent) and mentioned(canonical_numbers(ent), _canonical_text(text)):
        return True
    nums = plain_numbers(ent)       # 'the fourth quarter of 2023' is checked by its year; ordinals by number_mismatch
    if not nums:
        return False
    months = _calendar_words(fold(ent))
    # 'August 1969' is not grounded by '1969' alone: the month must match too
    if all(has_number(text_numbers, n) for n in nums) and months <= _source_calendar_words(text):
        return True
    return _in_numeric_date(nums, months, text)


def extract(nlp, text: str) -> Facts:
    doc = nlp(text)
    # Relative dates without numbers ("last year", "the century", "the fourth quarter") are too
    # paraphrase-prone to check; an ordinal is compared by number_mismatch instead.
    ents = [(e.text, e.label_) for e in doc.ents
            if e.label_ in FACT_ENTITY_LABELS and not (e.label_ == "DATE" and not plain_numbers(e.text))]
    found = [q for q in quantities(text) if not q.ordinal]
    return Facts({q.value for q in found}, ents, {q.value for q in found if q.decrease})


def conflicts(nlp, claim_text: str, claim: Facts, related: list[str], source_lower: str) -> list[str]:
    """Ways the claim disagrees with closely related evidence (related[0] is the best sentence)."""
    ev_text = " ".join(related)
    ev = extract(nlp, ev_text)
    source_lower, ev_lower = fold(source_lower), fold(ev_text)
    out = []

    mismatch = number_mismatch(claim_text, related[0] if quantities(related[0]) else ev_text, ev_text)
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


def canonical_numbers(text: str) -> str:
    """The text with every number in plain digits: '$2.1 billion' and '$2,100 million' -> '$2100000000',
    '1,280.5' -> '1280.5', 'twenty-one' -> '21', 'third' -> '3rd'.

    Two texts that differ only in how they write equal numbers then read the same to the NLI model.
    """
    t, found = _scan(text)
    out, last = [], 0
    for start, end, q in sorted(found, key=lambda f: f[0]):
        out += [t[last:start], _digits(q, t[start:end])]
        last = end
    return "".join(out) + t[last:]


@functools.lru_cache(maxsize=8)
def _canonical_text(text: str) -> str:
    """canonical_numbers() of a (folded) source text, cached: every claim is grounded against the same text."""
    return fold(canonical_numbers(text))


def _digits(q: Quantity, written: str) -> str:
    sign = "-" if q.value < 0 else ""
    currency = written.lstrip("-")[:1] if written.lstrip("-")[:1] in "$€£" else ""
    n = abs(q.value)
    digits = f"{n:.15g}"
    if q.ordinal:
        suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(int(n) % 10, "th")
        digits += suffix
    return sign + currency + digits


def _values(q: Quantity) -> set[float]:
    """What a number may stand for: 'fell 3.2%' is also the '-3.2%' a source may print."""
    return {q.value, -q.value} if q.decrease else {q.value}


def _band(value: float) -> int | None:
    return math.floor(math.log10(abs(value))) if value else None


def _same_kind(q: Quantity, e: Quantity) -> bool:
    """Both count the same thing. Two bare numbers ('3 million' vs '2,000,000') are taken to."""
    return q.ordinal == e.ordinal and bool((q.kinds & e.kinds) or not (q.kinds or e.kinds))


def _lone_partner(q: Quantity, ev_q: list[Quantity]) -> list[Quantity]:
    """The one evidence number a claim number must mean when only one side names its unit.

    '3.2 million units' vs 'reached 2.9 million': the unit is implicit on one side. This only holds for
    a plain count noun (never money, %, a time, a year or a day: '14 percent' does not stand in for
    '$412 million'), for numbers of the same order of magnitude, and when exactly one such number exists.
    """
    def fits(e: Quantity) -> bool:
        return (e.ordinal == q.ordinal and bool(e.kinds) != bool(q.kinds)
                and not (e.kinds | q.kinds) & (_STRONG_KINDS | _LABEL_NOUNS)
                and _band(e.value) is not None and _band(e.value) == _band(q.value))
    partners = [e for e in ev_q if fits(e)]
    return partners if len(partners) == 1 else []


def number_mismatch(claim_text: str, evidence: str, context: str = "") -> str | None:
    """A claim number that differs from every evidence number counting the same thing.

    Numbers are compared with the single best-matching sentence: it states the same fact. They count the
    same thing when they share a kind ('300 workers' and '1,500 people' both count people), when both are
    bare, or when one is bare and is the only same-sized number the other could mean (_lone_partner).
    A claim number with no such partner ('14 percent' against a sentence about '$412 million', or a claim
    that joins two sentences) is no conflict; novel() still checks that it appears somewhere in the sources.

    The lone-partner guess is the weakest of these rules, so it is dropped when `context` (all the related
    evidence) states the claim's number with the claim's own unit ('has 30 floors' against 'when he was 45'
    beside 'The building has 30 floors'), or when the partner is a number the claim states itself ('ranked
    15 ... opened 12 stores' against 'ranked 15'). The same number without that unit elsewhere ('1.4 million
    visitors in 2019' against '1.1 million in 2019 ... in 2018 attendance was 1.4 million') keeps the guess.
    A shared-kind mismatch always stays: the plant that 'opened with 300 workers' and 'today employs 1,500
    people' still contradicts 'today 300 workers'.
    """
    ev_q = quantities(evidence)
    ev_values = set().union(*(_values(e) for e in ev_q))
    claim_q = quantities(claim_text)
    claim_values = set().union(*(_values(q) for q in claim_q))
    context_q = quantities(context) if context else []
    claim_vals, ev_vals = set(), set()
    for q in claim_q:
        if any(has_number(ev_values, v) for v in _values(q)):
            continue
        partners = [e for e in ev_q if _same_kind(q, e)]
        stated_with_unit = any(q.ordinal == c.ordinal and q.kinds & c.kinds
                               and any(has_number(_values(c), v) for v in _values(q)) for c in context_q)
        if not partners and not stated_with_unit:
            partners = [e for e in _lone_partner(q, ev_q)
                        if not any(has_number(claim_values, v) for v in _values(e))]
        same_kind = {e.value for e in partners}
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
    return _flipped(e, c) or _flipped(c, e) or _exclusion_flipped(e, c) or _exclusion_flipped(c, e)


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


_ARGUMENT_DEPS = {"conj", "dobj", "prep", "pobj", "pcomp", "dative"}


def _exclusions(doc) -> list[tuple[object, object]]:
    """'covers dental care but not vision care' -> (the predicate 'covers', the excluded argument 'care').

    Here the negation sits on an argument, not on the verb. The predicate is found by walking up through
    objects, prepositions and conjuncts, and must be what the sentence asserts.
    """
    out = []
    for t in doc:
        if t.dep_ != "neg" or t.head.pos_ in ("VERB", "AUX"):
            continue
        pred = t.head
        while pred.dep_ in _ARGUMENT_DEPS:
            pred = pred.head
        if pred is not t.head and _is_main(pred):
            out.append((pred, t.head))
    return out


def _content_lemmas(tokens) -> set[str]:
    return {w.lemma_.lower() for w in tokens if w.pos_ in _CONTENT_POS and w.dep_ != "neg"}


def _exclusion_flipped(negated_doc, other_doc) -> bool:
    """An argument excluded in `negated_doc` ('but not vision care') is asserted in `other_doc`'s main clause.

    The argument is all of its content words, so 'covers dental care' does not assert the excluded
    'vision care' through the shared 'care'.
    """
    # what the other text excludes itself is not asserted there ('dental care but not vision care' again)
    excluded_there = {w.i for _, arg in _exclusions(other_doc) for w in arg.subtree}
    for pred, arg in _exclusions(negated_doc):
        words, lemma = _content_lemmas(arg.subtree), pred.lemma_.lower()
        for tok in other_doc:
            if tok.lemma_.lower() != lemma or not _is_main(tok) or any(ch.dep_ == "neg" for ch in tok.children):
                continue
            asserted = _content_lemmas(w for w in tok.subtree if w.i not in excluded_there and w.i != tok.i)
            if words and words <= asserted:
                return True
    return False


def _is_light(tok) -> bool:
    return tok.pos_ == "AUX" or tok.lemma_.lower() in _LIGHT_VERBS


def _negated_predicates(doc) -> list:
    """For 'X is not Y' the negation sits on the copula: the predicate is Y, not 'be'.

    A verb coordinated by 'or' / 'nor' under the same negation is negated too: in "did not raise
    prices or cut jobs" both 'raise' and 'cut' are denied.
    """
    out = []
    for t in doc:
        if t.dep_ != "neg":
            continue
        head = t.head
        preds = [ch for ch in head.children if ch.dep_ in _PREDICATE_DEPS] if _is_light(head) else []
        out += preds or [head]
        frontier = [head]
        while frontier:
            for conj in frontier.pop().children:
                if _shares_negation(conj):
                    out.append(conj)
                    frontier.append(conj)
    return out


def _shares_negation(conj) -> bool:
    """A conjunct verb with no subject, auxiliary or negation of its own, joined by 'or' / 'nor'.

    'and' does not carry the negation: "do not produce insulin and require therapy" affirms 'require'.
    """
    own = {c.dep_ for c in conj.children}
    if conj.dep_ != "conj" or conj.pos_ not in ("VERB", "AUX") or \
            own & {"nsubj", "nsubjpass", "expl", "aux", "auxpass", "neg"}:
        return False
    head = conj.head
    ccs = ([c for c in head.children if c.dep_ == "cc" and head.i < c.i < conj.i]
           or [c for c in conj.children if c.dep_ == "cc" and c.i > conj.i])
    return bool(ccs) and ccs[-1].lower_ in ("or", "nor")


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


_LOWER_ABBR_RE = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)*)([kmb])\b")


@functools.lru_cache(maxsize=8)
def _source_values(source_text: str) -> frozenset[float]:
    """Every value the sources state, read before lower-casing so '5M' stays 5 million.

    Text that is already all lower-case has lost that case, so a glued '5m' there is read both as 5 and
    as 5 million: grounding a claim's '5M users' must not depend on how the caller cased the sources.
    Cached: every claim of a response is checked against the same sources.
    """
    values = set().union(*(_values(q) for q in quantities(source_text)))
    if source_text == source_text.lower():
        upper = _LOWER_ABBR_RE.sub(lambda m: m[1] + m[2].upper(), source_text)
        values |= set().union(*(_values(q) for q in quantities(upper)))
    return frozenset(values)


def novel(claim: Facts, source_text: str) -> list[str]:
    """Numbers and names in the claim that appear in no source at all. Ordinals are not checked here."""
    src_nums = _source_values(source_text)
    source_lower = fold(source_text)
    out = [fmt_numbers({n}) for n in claim.numbers
           if not has_number(src_nums, n) and not (n in claim.decreasing and has_number(src_nums, -n))]
    for ent, label in claim.entities:
        if not grounded(ent, label, source_lower, src_nums):
            out.append(f"'{ent}'")
    return out
