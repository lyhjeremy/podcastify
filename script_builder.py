"""
script_builder.py - turn a block of source text into a two-host podcast script.
==============================================================================
Two modes:

    "rule"  (default, no API key)  A structured two-voice reading: the main host
            narrates the source in short segments, and a co-host drops in natural
            reactions / transition questions between segments. Faithful to the
            source text, just dramatised for two voices.

    "llm"   (optional, needs ANTHROPIC_API_KEY)  Rewrites the source into a genuine
            back-and-forth conversation with the Claude API.

Both return a list of (speaker, text) turns, where speaker is "A" or "B".
"""
from __future__ import annotations
import os
import re
import json
import random
import urllib.request


# ---- co-host interjections, rotated so they don't repeat too soon -----------
_REACTIONS = [
    "Okay, that's fascinating.",
    "Huh, I did not know that.",
    "Wait, can you unpack that a little?",
    "So what does that actually mean for the reader?",
    "Interesting - and then what happens?",
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
    "Right - so it continues like this.",
    "So here's what comes next.",
    "Stay with me, it builds.",
]


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    # split on sentence-ending punctuation, keeping the punctuation
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [p.strip() for p in parts if p.strip()]


def segment(text: str, sentences_per_segment: int = 3) -> list[str]:
    """Group the source into bite-sized narration chunks."""
    sents = split_sentences(text)
    segs, cur = [], []
    for s in sents:
        cur.append(s)
        if len(cur) >= sentences_per_segment:
            segs.append(" ".join(cur))
            cur = []
    if cur:
        segs.append(" ".join(cur))
    return segs


def build_rule_based(text: str, title: str = "today's reading",
                     sentences_per_segment: int = 3) -> list[tuple[str, str]]:
    segs = segment(text, sentences_per_segment)
    turns: list[tuple[str, str]] = []
    # intro
    turns.append(("B", f"Welcome to Podcastify. Today we're getting into {title}."))
    turns.append(("A", "That's right. Let's dive in."))

    reactions = _REACTIONS[:]
    random.shuffle(reactions)
    ri = 0
    for i, seg in enumerate(segs):
        turns.append(("A", seg))
        # co-host interjects every couple of segments (not after the last)
        if i < len(segs) - 1 and i % 2 == 1:
            turns.append(("B", reactions[ri % len(reactions)]))
            ri += 1
            turns.append(("A", random.choice(_BRIDGES_A)))
    # outro
    turns.append(("B", "That's a great place to wrap up. Thanks for listening to Podcastify."))
    turns.append(("A", "See you next time."))
    return turns


# ---- optional Claude-powered conversational rewrite -------------------------
def build_with_llm(text: str, title: str = "today's reading",
                   model: str = "claude-opus-4-8") -> list[tuple[str, str]]:
    """Rewrite the source into a natural two-host conversation using the Claude API.

    Requires the ANTHROPIC_API_KEY environment variable. Falls back to the
    rule-based builder if no key is present.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return build_rule_based(text, title)

    prompt = (
        "Turn the following source text into a lively, accurate two-host podcast "
        "conversation between host A (warm narrator) and host B (curious co-host). "
        "Stay faithful to the content; do not invent facts. Return ONLY a JSON array "
        "of objects like {\"speaker\":\"A\"|\"B\",\"text\":\"...\"}.\n\n"
        f"TITLE: {title}\n\nSOURCE:\n{text[:12000]}"
    )
    body = json.dumps({
        "model": model,
        "max_tokens": 4000,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    raw = data["content"][0]["text"]
    m = re.search(r"\[.*\]", raw, re.S)
    arr = json.loads(m.group(0))
    return [(t.get("speaker", "A"), t.get("text", "")) for t in arr if t.get("text")]


def build_script(text: str, title: str = "today's reading", mode: str = "rule",
                 **kw) -> list[tuple[str, str]]:
    if mode == "llm":
        return build_with_llm(text, title)
    return build_rule_based(text, title, **kw)


def script_to_markdown(turns: list[tuple[str, str]], names=("Alex", "Sam")) -> str:
    a, b = names
    out = ["# Podcast transcript\n"]
    for spk, txt in turns:
        out.append(f"**{a if spk == 'A' else b}:** {txt}\n")
    return "\n".join(out)
