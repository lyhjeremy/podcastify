"""
script_builder.py - turn a source text into a two-host podcast script.
=====================================================================
Two modes, both returning a list of line dicts:

    {"speaker": "A" | "B", "text": str, "section": int, "claim": bool, "src": ["S12", ...]}

    "rule"  (default, no LLM)  A two-voice reading: Alex reads the source in
            three-sentence segments, and Sam drops in a reaction from a fixed pool
            after every second one. Every narration line is the source verbatim,
            so it cites its own sentences.

    "llm"   The grounded pipeline in pipeline.py and grounding.py: map the whole
            source to cited key points, reduce to an outline, write the
            conversation with citations, then check and repair every line.
            Runs through the local `claude -p` CLI, or the Anthropic API when
            ANTHROPIC_API_KEY is set.
"""
from __future__ import annotations

import random

import textspans

# ---- co-host interjections, rotated so they don't repeat too soon -----------
_REACTIONS = [
    "Okay, that's fascinating.",
    "Huh, I did not know that.",
    "Wait, can you unpack that a little?",
    "So what does that actually mean for the reader?",
    "Interesting, and then what happens?",
    "Love that. Keep going.",
    "That's a great point.",
    "Mm, that raises a question for me.",
    "Right, that makes sense.",
    "Oh, I like where this is going.",
    "Let me make sure I'm following.",
    "And why does that matter?",
]
_BRIDGES_A = [
    "Good question. Here's the next part.",
    "Exactly. Let me read on.",
    "Right, so it continues like this.",
    "So here's what comes next.",
    "Stay with me, it builds.",
]


def _line(speaker, text, claim=False, src=None, section=0):
    return {"speaker": speaker, "text": text, "section": section, "claim": claim, "src": src or []}


def build_rule_based(text: str, title: str = "today's reading", sentences_per_segment: int = 3,
                     max_words: int = 0, seed: int | None = None) -> list[dict]:
    """Two-voice reading. max_words stops the reading once that many source words are read."""
    rng = random.Random(seed)
    sents = textspans.sentences(text)
    segs = [sents[i:i + sentences_per_segment] for i in range(0, len(sents), sentences_per_segment)]
    lines = [_line("B", f"Welcome to Podcastify. Today we're getting into {title}."),
             _line("A", "That's right. Let's dive in.")]
    reactions = _REACTIONS[:]
    rng.shuffle(reactions)
    ri = words = 0
    for i, seg in enumerate(segs):
        body = " ".join(text[s["start"]:s["end"]] for s in seg)
        lines.append(_line("A", body, True, [s["id"] for s in seg]))
        words += len(body.split())
        if max_words and words >= max_words:
            break
        if i < len(segs) - 1 and i % 2 == 1:
            lines.append(_line("B", reactions[ri % len(reactions)]))
            ri += 1
            lines.append(_line("A", rng.choice(_BRIDGES_A)))
    lines.append(_line("B", "That's a great place to wrap up. Thanks for listening to Podcastify."))
    lines.append(_line("A", "See you next time."))
    return lines


def build_llm(text: str, meta: dict, minutes: float = 6, check: bool = True, model=None,
              log=print) -> dict:
    """The grounded LLM pipeline. Returns an episode dict (outline, lines, checks)."""
    import grounding
    import pipeline

    sents, points, sections = pipeline.build_outline(text, meta, minutes, model, log)
    log("  script pass")
    lines, shown = pipeline.script_pass(text, sents, points, sections, meta, minutes, model)
    log(f"  script: {len(lines)} lines, {sum(len(l['text'].split()) for l in lines):,} words")
    draft = [dict(l) for l in lines]
    if check:
        lines = grounding.check_script(lines, text, sents, shown, repair=True, model=model, log=log,
                                       meta=meta)
    return {"sents": sents, "points": points, "sections": sections, "draft": draft, "lines": lines,
            "shown": shown, "checked": check}


def build_script(text: str, title: str = "today's reading", mode: str = "rule", **kw) -> list[dict]:
    if mode == "llm":
        meta = kw.pop("meta", {"title": title})
        ep = build_llm(text, meta, **kw)
        return [l for l in ep["lines"] if not l.get("dropped")]
    return build_rule_based(text, title, **kw)


def spoken(lines: list[dict]) -> list[dict]:
    return [l for l in lines if not l.get("dropped")]


def script_to_markdown(lines: list[dict], names=("Alex", "Sam"), sections=None,
                       text: str | None = None, sents: list[dict] | None = None) -> str:
    """Transcript with citations and check verdicts. Dropped lines are listed struck out."""
    a, b = names
    idx = textspans.index(sents) if sents else {}
    out = ["# Podcast transcript\n"]
    cur = None
    for ln in lines:
        if sections and ln.get("section") != cur and ln.get("section", 0) < len(sections):
            cur = ln["section"]
            out.append(f"## {sections[cur]['title']}\n")
        who = a if ln["speaker"] == "A" else b
        cite = ""
        if ln.get("src"):
            spans = [f"{s} @{idx[s]['start']}-{idx[s]['end']}" if s in idx else s for s in ln["src"]]
            cite = f" [{'; '.join(spans)}]"
        chk = ln.get("check") or {}
        verdict = ""
        if chk.get("first") and chk["first"] != "no_claim":
            verdict = f" ({chk['first']}" + (f", {chk['action']}" if chk.get("action") != "kept" else "") + ")"
        if ln.get("dropped"):
            out.append(f"~~**{who}:** {chk.get('original', ln['text'])}~~ (dropped: {chk.get('why', '')})\n")
            continue
        if chk.get("action") == "rewritten":
            out.append(f"**{who}:** {ln['text']}{cite}{verdict}  \n"
                       f"  was: {chk['original']}\n")
            continue
        out.append(f"**{who}:** {ln['text']}{cite}{verdict}\n")
    return "\n".join(out)
