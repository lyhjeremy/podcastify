#!/usr/bin/env python3
"""Download and clean the four eval sources into ~/data-cache/podcastify/sources/.

    python eval/fetch_sources.py

Each cleaned file is plain UTF-8 with one paragraph per block (hard line wraps
joined, Gutenberg boilerplate removed). Character offsets in every transcript
and outline refer to these cleaned files, so the cleaning has to be repeatable.
The Wikipedia article is pinned to a revision id.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

CACHE = os.path.expanduser("~/data-cache/podcastify/sources")
HERE = os.path.dirname(os.path.abspath(__file__))

SOURCES = {
    "alice": {
        "title": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
        "kind": "novel, whole book",
        "url": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
        "license": "Public domain (Project Gutenberg eBook #11)",
        "start": r"^CHAPTER I\.\s*$", "skip_toc": True, "end": r"^THE END\s*$",
    },
    "civil": {
        "title": "On the Duty of Civil Disobedience",
        "author": "Henry David Thoreau",
        "kind": "essay",
        "url": "https://www.gutenberg.org/cache/epub/71/pg71.txt",
        "license": "Public domain (Project Gutenberg eBook #71)",
        "start": r"^I heartily accept the motto", "end": r"^\*\*\* END OF THE PROJECT",
    },
    "origin3": {
        "title": "On the Origin of Species, Chapter III: Struggle for Existence",
        "author": "Charles Darwin",
        "kind": "nonfiction chapter",
        "url": "https://www.gutenberg.org/cache/epub/1228/pg1228.txt",
        "license": "Public domain (Project Gutenberg eBook #1228, first edition)",
        "start": r"^CHAPTER III\.\s*$", "skip_toc": True, "end": r"^CHAPTER IV\.\s*$",
    },
    "greatstink": {
        "title": "Great Stink",
        "author": "Wikipedia contributors",
        "kind": "encyclopedia article",
        "url": "https://en.wikipedia.org/w/index.php?title=Great_Stink&oldid=1381900566",
        "api": "https://en.wikipedia.org/w/api.php?action=query&prop=extracts|revisions&rvprop=ids|timestamp&explaintext=1&revids=1381900566&format=json&formatversion=2",
        "license": "CC BY-SA 4.0, Wikipedia revision 1381900566 (2026-10-25)",
    },
}


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "podcastify-eval/1.0 (lyhjeremy.github.io/podcastify)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def unwrap(text: str) -> str:
    """Join hard-wrapped lines into paragraphs; drop _italic_ markers; em dash to "--"."""
    paras = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    out = []
    for p in paras:
        p = re.sub(r"\s*\n\s*", " ", p).strip()
        p = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", p)
        # Em dashes become "--", the old Project Gutenberg plain-text convention,
        # so quoted source text can sit on the site under the no-em-dash rule.
        p = p.replace("\u2014", "--")
        if p:
            out.append(p)
    return "\n\n".join(out) + "\n"


def gutenberg(spec: dict) -> str:
    raw = _get(spec["url"])
    lines = raw.replace("\r\n", "\n").split("\n")
    starts = [i for i, l in enumerate(lines) if re.match(spec["start"], l)]
    # Some books list the chapter heading in a table of contents first.
    s = starts[-1] if spec.get("skip_toc") and len(starts) > 1 else starts[0]
    e = next(i for i in range(s + 1, len(lines)) if re.match(spec["end"], lines[i]))
    return unwrap("\n".join(lines[s:e]))


def wikipedia(spec: dict) -> str:
    d = json.loads(_get(spec["api"]))
    text = d["query"]["pages"][0]["extract"]
    text = re.split(r"\n== See also ==\n", text)[0]
    # Section headings become their own short paragraphs.
    text = re.sub(r"\n=+ (.*?) =+\n", r"\n\n\1\n\n", text)
    return unwrap(text)


def main() -> None:
    os.makedirs(CACHE, exist_ok=True)
    meta = {}
    for key, spec in SOURCES.items():
        text = wikipedia(spec) if "api" in spec else gutenberg(spec)
        path = os.path.join(CACHE, f"{key}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        m = {k: v for k, v in spec.items() if k in ("title", "author", "kind", "url", "license")}
        m["chars"] = len(text)
        m["words"] = len(text.split())
        meta[key] = m
        print(f"{key:11s} {m['chars']:>8,} chars {m['words']:>7,} words  {path}")
    with open(os.path.join(HERE, "sources.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
