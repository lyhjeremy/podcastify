"""
textspans.py - split a source into numbered sentences with character offsets.
=============================================================================
Every source span in Podcastify is a sentence of the source text, with an id
("S41") and its [start, end) character offsets into the text as read. The
outline, the script and the grounding check all cite these ids, so a claim can
be traced back to the exact characters it rests on.
"""
from __future__ import annotations

import re

# A sentence ends at . ! or ? (plus any closing quotes or brackets), when the
# next non-space character opens a new sentence: a capital, digit or opening quote.
_END = re.compile(r"[.!?]+[’”'\")\]]*(?=\s+[‘“'\"(\[_]*[A-Z0-9])")
_ABBR = {"mr", "mrs", "ms", "dr", "st", "sir", "jr", "sr", "vs", "etc", "no", "vol",
         "fig", "ch", "mt", "gen", "col", "capt", "rev", "prof", "i.e", "e.g", "c"}


def sentences(text: str, max_len: int = 900) -> list[dict]:
    """Return [{"id": "S1", "start": int, "end": int}, ...] covering the text.

    Paragraph breaks always end a sentence. Very long sentences (Carroll and
    Thoreau both write 150-word ones) are split at a semicolon or colon so a
    span stays short enough to read in a pop-up.
    """
    out: list[dict] = []
    for pm in re.finditer(r"[^\n]+", text):
        p0, para = pm.start(), pm.group(0)
        cut = 0
        bounds = []
        for m in _END.finditer(para):
            word = re.findall(r"([A-Za-z.]+)\.$", para[cut:m.start() + 1])
            if word and word[-1].lower().rstrip(".") in _ABBR and m.group(0) == ".":
                continue
            bounds.append(m.end())
        bounds.append(len(para))
        for b in bounds:
            seg = para[cut:b]
            for s0, s1 in _split_long(seg, max_len):
                lo = cut + s0 + (len(seg[s0:s1]) - len(seg[s0:s1].lstrip()))
                hi = cut + s1 - (len(seg[s0:s1]) - len(seg[s0:s1].rstrip()))
                if hi > lo:
                    out.append({"start": p0 + lo, "end": p0 + hi})
            cut = b
    for i, s in enumerate(out):
        s["id"] = f"S{i + 1}"
    return out


def _split_long(seg: str, max_len: int) -> list[tuple[int, int]]:
    if len(seg) <= max_len:
        return [(0, len(seg))]
    parts, start = [], 0
    for m in re.finditer(r"[;:]\s+", seg):
        if m.end() - start >= max_len * 0.4:
            parts.append((start, m.end()))
            start = m.end()
    parts.append((start, len(seg)))
    return parts


def chunks(sents: list[dict], max_chars: int = 10000) -> list[list[dict]]:
    """Group consecutive sentences into chunks of at most ~max_chars."""
    out, cur, size = [], [], 0
    for s in sents:
        n = s["end"] - s["start"]
        if cur and size + n > max_chars:
            out.append(cur)
            cur, size = [], 0
        cur.append(s)
        size += n + 1
    if cur:
        out.append(cur)
    return out


def span_text(text: str, sent_index: dict, ids: list[str]) -> str:
    return " ".join(text[sent_index[i]["start"]:sent_index[i]["end"]] for i in ids if i in sent_index)


def index(sents: list[dict]) -> dict:
    return {s["id"]: s for s in sents}
