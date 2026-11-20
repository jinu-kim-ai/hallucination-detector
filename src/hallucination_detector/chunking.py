"""Cutting over-long text into bounded pieces, with character offsets kept exact.

Used on both sides: source documents (sources.py) and the response (claims.py). Bounded pieces keep
memory and run time linear in the length of the input, whatever it looks like.
"""
from __future__ import annotations

import re

# spaCy's parser needs a lot of memory on very long texts and refuses anything over nlp.max_length.
# Parsing in pieces of this many characters keeps peak memory flat; sentences are found per piece.
PARSE_CHUNK_CHARS = 100_000


def parse_limit(nlp) -> int:
    return min(getattr(nlp, "max_length", 1_000_000), PARSE_CHUNK_CHARS)


def chunks(text: str, limit: int) -> list[tuple[int, str]]:
    """(offset, chunk) pieces of at most `limit` characters, cut at the best boundary available."""
    out, start = [], 0
    while len(text) - start > limit:
        cut = cut_point(text, start, start + limit)
        out.append((start, text[start:cut]))
        start = cut
    out.append((start, text[start:]))
    return out


def cut_point(text: str, lo: int, hi: int) -> int:
    """Last paragraph break, line break, sentence end or space in the back half of text[lo:hi]."""
    mid = lo + (hi - lo) // 2                       # cutting in the back half keeps the chunk count linear
    window = text[mid:hi]
    for pattern in (r"\n\s*\n", r"\n", r"[.!?][\"')\]]*\s", r"\s"):
        last = None
        for last in re.finditer(pattern, window):
            pass
        if last is not None:
            return mid + last.end()
    return hi                                       # one enormous token: a hard cut is all that is left


def trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def split_long(text: str, start: int, end: int, max_words: int) -> list[tuple[int, int]]:
    """Cut an over-long sentence at semicolons / commas into pieces of at most `max_words` words."""
    pieces, piece_start, words = [], start, 0
    for m in re.finditer(r"\S+", text[start:end]):
        words += 1
        token_end = start + m.end()
        if words >= max_words // 2 and m.group().endswith((";", ",")) or words >= max_words:
            pieces.append(trim(text, piece_start, token_end))
            piece_start, words = token_end, 0
    if piece_start < end and text[piece_start:end].strip():
        pieces.append(trim(text, piece_start, end))
    return pieces
