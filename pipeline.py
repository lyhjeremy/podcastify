"""
pipeline.py - the LLM path: map, reduce, script, each step cited to source spans.
=================================================================================
1. map     Every chunk of the whole source (about 10,000 characters each) goes to
           the model with its sentences numbered. It returns key points, each
           citing the sentence ids it comes from.
2. reduce  The model sees every key point and picks what an episode of the
           target length covers, grouped into sections in source order.
3. script  The model writes the host and co-host conversation from the outline
           and the cited sentences. Every line that says anything about the
           source carries the sentence ids it rests on.

Each step's reply is cached on disk by a hash of its prompt, so a rerun after a
failure picks up where it stopped. The grounding check lives in grounding.py.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os

import llm
import textspans

CACHE_DIR = os.environ.get("PODCASTIFY_CACHE", os.path.expanduser("~/.cache/podcastify"))
WPM = 155  # edge-tts default voices measured 150 to 165 words a minute in this repo's runs


def cached(tag: str, prompt: str, system: str, model: str | None = None):
    """complete_json with a disk cache keyed on (model, system, prompt)."""
    key = hashlib.sha256(f"{model}\n{system}\n{prompt}".encode()).hexdigest()[:20]
    path = os.path.join(CACHE_DIR, f"{tag}-{key}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    data = llm.complete_json(prompt, system, model=model)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    return data


def numbered(text: str, sents: list[dict]) -> str:
    return "\n".join(f"[{s['id']}] {text[s['start']:s['end']]}" for s in sents)


# ---- 1. map ------------------------------------------------------------------
MAP_SYSTEM = ("You extract key points from a source text for a faithful audio summary. "
              "You only report what the text says. Reply with JSON only.")
MAP_PROMPT = """This is part {i} of {n} of "{title}". Its sentences are numbered.

List the key points of this part: the main events (for a story), claims and reasons (for an
argument) or facts and figures (for a factual text). Write about one point per 1,500 characters,
more if the part is dense. Each point is one plain sentence that the cited sentences fully support,
and cites the ids of every sentence it rests on (usually one to four). Give each point an importance
from 1 (detail) to 3 (essential to the whole work).

Reply as JSON: {{"points": [{{"point": "...", "sents": ["S12", "S13"], "importance": 2}}]}}

PART {i}:
{body}
"""


def map_pass(text: str, sents: list[dict], title: str, model=None, workers: int = 4) -> list[dict]:
    parts = textspans.chunks(sents)
    valid = {s["id"] for s in sents}

    def one(i_part):
        i, part = i_part
        prompt = MAP_PROMPT.format(i=i + 1, n=len(parts), title=title, body=numbered(text, part))
        data = cached("map", prompt, MAP_SYSTEM, model)
        out = []
        for p in data.get("points", []):
            ids = [x for x in p.get("sents", []) if x in valid]
            if p.get("point") and ids:
                out.append({"point": p["point"], "sents": ids, "importance": int(p.get("importance", 2)),
                            "part": i})
        return out

    with cf.ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(one, enumerate(parts)))
    points = [p for r in results for p in r]
    for k, p in enumerate(points):
        p["id"] = f"P{k + 1}"
    return points


# ---- 2. reduce -----------------------------------------------------------------
REDUCE_SYSTEM = ("You plan a faithful two-host audio episode about a source text. "
                 "Reply with JSON only.")
REDUCE_PROMPT = """Below are the key points extracted from the whole of "{title}" ({kind}), in source order,
with an importance from 1 to 3. Plan an episode of about {minutes} minutes (about {words} spoken words).

Pick about {n_points} points that together give a listener the whole arc or argument of the work,
from beginning to end, not just its opening. Prefer importance 3, keep the source order, and group
them into {n_sections} sections with short plain titles (two to five words).

Reply as JSON: {{"sections": [{{"title": "...", "points": ["P1", "P4"]}}]}}

POINTS:
{points}
"""


def reduce_pass(points: list[dict], title: str, kind: str, minutes: float, model=None) -> list[dict]:
    words = int(minutes * WPM)
    n_points = max(4, round(minutes * 2.2))
    n_sections = 3 if minutes < 5 else 4 if minutes < 9 else 5
    listing = "\n".join(f"{p['id']} (part {p['part'] + 1}, importance {p['importance']}): {p['point']}"
                        for p in points)
    prompt = REDUCE_PROMPT.format(title=title, kind=kind, minutes=minutes, words=words,
                                  n_points=n_points, n_sections=n_sections, points=listing)
    data = cached("reduce", prompt, REDUCE_SYSTEM, model)
    known = {p["id"] for p in points}
    sections = []
    for s in data.get("sections", []):
        ids = [x for x in s.get("points", []) if x in known]
        if ids:
            sections.append({"title": s.get("title", f"Part {len(sections) + 1}"), "points": ids})
    return sections


# ---- 3. script -----------------------------------------------------------------
SCRIPT_SYSTEM = ("You write scripts for a two-host audio show that explains one source text. "
                 "Every fact in the script comes from the source sentences you are given. "
                 "Reply with JSON only.")
SCRIPT_PROMPT = """Write a two-host conversation about "{title}" by {author} ({kind}), about {words} words in total.

Alex (speaker "A") has read it closely and walks through it. Sam (speaker "B") is curious, asks the
questions a listener would ask, reacts, and sometimes sums up. Write for the ear: short sentences,
natural turns, most turns under 45 words, no em dashes. Follow the sections in order; Alex may quote a short phrase
from the source when it is vivid. Open with one or two lines naming the work and its author, and
close with one or two lines.

Grounding rules, which matter more than style:
- Use only what the cited source sentences below say. Do not add background, dates, biography,
  later history or interpretation that the sentences do not state.
- Mark a line "claim": true if it states or presupposes anything about the source's content
  (an event, a character's action, an argument, a fact, a number), including a question or a summary
  that does. Give it "src": the ids of the sentences that support it, taken only from the ones listed.
- Pure reactions, transitions and questions that assert nothing are "claim": false with "src": [].

Reply as JSON: {{"lines": [{{"section": 0, "speaker": "A", "text": "...", "claim": true, "src": ["S12"]}}]}}
Sections are numbered from 0 in the order given.

OUTLINE AND SOURCE SENTENCES:
{outline}
"""


def outline_evidence(text: str, sents: list[dict], points: list[dict], sections: list[dict],
                     context: int = 1) -> tuple[str, dict]:
    """Render the outline with each point's cited sentences plus `context` neighbours.
    Returns (rendered text, {section index: [sentence ids shown]})."""
    pos = {s["id"]: i for i, s in enumerate(sents)}
    by_id = {p["id"]: p for p in points}
    blocks, shown = [], {}
    for k, sec in enumerate(sections):
        lines = [f"## Section {k}: {sec['title']}"]
        ids_here: list[str] = []
        for pid in sec["points"]:
            p = by_id[pid]
            idx = sorted(pos[x] for x in p["sents"])
            near = sorted({j for i in idx for j in range(max(0, i - context), min(len(sents), i + context + 1))})
            lines.append(f"Point: {p['point']}")
            for j in near:
                s = sents[j]
                lines.append(f"  [{s['id']}] {text[s['start']:s['end']]}")
                ids_here.append(s["id"])
        shown[k] = list(dict.fromkeys(ids_here))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks), shown


def script_pass(text: str, sents: list[dict], points: list[dict], sections: list[dict], meta: dict,
                minutes: float, model=None) -> tuple[list[dict], dict]:
    body, shown = outline_evidence(text, sents, points, sections)
    prompt = SCRIPT_PROMPT.format(title=meta["title"], author=meta.get("author", "an unknown author"),
                                  kind=meta.get("kind", "text"), words=int(minutes * WPM), outline=body)
    data = cached("script", prompt, SCRIPT_SYSTEM, model)
    valid = {s["id"] for s in sents}
    lines = []
    for ln in data.get("lines", []):
        txt = (ln.get("text") or "").replace("\u2014", ", ").strip()
        if not txt:
            continue
        src = [x for x in ln.get("src", []) if x in valid]
        lines.append({"speaker": "B" if ln.get("speaker") == "B" else "A", "text": txt,
                      "section": int(ln.get("section", 0)), "claim": bool(ln.get("claim")),
                      "src": src})
    return lines, shown


def build_outline(text: str, meta: dict, minutes: float, model=None, log=print):
    sents = textspans.sentences(text)
    log(f"  {len(sents):,} sentences in {len(textspans.chunks(sents))} chunks; map pass")
    points = map_pass(text, sents, meta["title"], model)
    log(f"  {len(points)} key points; reduce pass for {minutes:g} minutes")
    sections = reduce_pass(points, meta["title"], meta.get("kind", "text"), minutes, model)
    log(f"  outline: {len(sections)} sections, {sum(len(s['points']) for s in sections)} points")
    return sents, points, sections
