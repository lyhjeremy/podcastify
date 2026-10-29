#!/usr/bin/env python3
"""Write the frozen key-point list for each eval source, before any episode exists.

    python eval/make_keypoints.py

One `claude -p` call per source with Opus (the episode pipeline uses Sonnet), with a
prompt that knows nothing about podcasts or outlines. The lists were reviewed by
hand against the sources, then committed before the first generation run; the
coverage score counts how many of these points each script mentions.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.expanduser("~/data-cache/podcastify/sources")

SYSTEM = "You are a careful reader who writes study notes. Reply with JSON only."
PROMPT = """Read the text below and list the 12 most important points a reader should take away from it.
For a story, cover the main events and characters in order; for an argument, the main claims and
reasons; for a factual text, the main facts, figures and causes. Each point is one plain sentence
that can be checked against the text. Do not add anything the text does not say.

Reply as JSON: {{"points": ["...", "..."]}}

TITLE: {title}

TEXT:
{text}
"""


def main() -> None:
    meta = json.load(open(os.path.join(HERE, "sources.json")))
    os.makedirs(os.path.join(HERE, "keypoints"), exist_ok=True)
    for key, m in meta.items():
        out = os.path.join(HERE, "keypoints", f"{key}.json")
        if os.path.exists(out):
            print(f"{key}: exists, frozen; not regenerating")
            continue
        text = open(os.path.join(CACHE, f"{key}.txt"), encoding="utf-8").read()
        data = llm.complete_json(PROMPT.format(title=m["title"], text=text), SYSTEM, model="opus")
        pts = [{"id": f"K{i + 1}", "point": p} for i, p in enumerate(data["points"])]
        json.dump({"source": key, "model": "opus (claude -p)", "points": pts},
                  open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"{key}: {len(pts)} points")


if __name__ == "__main__":
    main()
