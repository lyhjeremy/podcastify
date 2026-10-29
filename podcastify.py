#!/usr/bin/env python3
"""
podcastify.py - turn an EPUB or text file into a two-host podcast.
==================================================================

    python podcastify.py book.epub                   # rule mode: two-voice reading, no LLM
    python podcastify.py book.epub --mode llm        # grounded conversation via `claude -p`
    python podcastify.py book.epub --mode llm --minutes 10 -o book.m4b
    python podcastify.py article.txt --mode llm --script-only   # transcript + episode JSON only

Setup:
    pip install -r requirements.txt
    # ffmpeg and ffprobe on PATH
    # llm mode: the Claude CLI signed in (claude.com/claude-code), or ANTHROPIC_API_KEY set

This tool ships no books or audio. Point it at a file you own or a public-domain
title; output is for personal use.
"""
from __future__ import annotations
import os
import sys
import argparse
import json

import script_builder
import tts_engine

_AUTHOR = ""


def read_source(path: str) -> tuple[str, str]:
    """Return (title, text) from an .epub or .txt/.md/.html file."""
    title = os.path.splitext(os.path.basename(path))[0]
    global _AUTHOR
    ext = os.path.splitext(path)[1].lower()
    if ext == ".epub":
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
        book = epub.read_epub(path)
        try:
            t = book.get_metadata("DC", "title")
            if t:
                title = t[0][0]
            a = book.get_metadata("DC", "creator")
            if a:
                _AUTHOR = a[0][0]
        except Exception:
            pass
        chunks = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            chunks.append(soup.get_text(" ", strip=True))
        return title, "\n\n".join(chunks)
    # plain text / html / markdown
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        data = fh.read()
    if ext in (".html", ".htm"):
        from bs4 import BeautifulSoup
        data = BeautifulSoup(data, "html.parser").get_text(" ", strip=True)
    return title, data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Turn an EPUB or text file into a two-host podcast.")
    ap.add_argument("input", help="path to .epub, .txt, .md, or .html")
    ap.add_argument("-o", "--out", help="output audio path, .mp3 or .m4b (default: <name>.podcast.mp3)")
    ap.add_argument("--mode", choices=["rule", "llm"], default="rule",
                    help="rule = two-voice reading, no LLM; llm = grounded conversation via claude -p or the API")
    ap.add_argument("--minutes", type=float, default=8, help="llm mode: target episode length")
    ap.add_argument("--no-check", action="store_true", help="llm mode: skip the grounding check and repair")
    ap.add_argument("--model", default=None, help="llm mode: model alias or id (default: sonnet)")
    ap.add_argument("--title", help="title to announce (default: EPUB metadata or file name)")
    ap.add_argument("--author", default="", help="author to announce in llm mode")
    ap.add_argument("--kind", default="text", help="what the source is, e.g. 'novel' or 'essay' (llm mode)")
    ap.add_argument("--max-chars", type=int, default=0,
                    help="cap source length (0 = whole file); handy for a quick preview")
    ap.add_argument("--segment", type=int, default=3, help="rule mode: sentences per narration segment")
    ap.add_argument("--voice-a", default=tts_engine.DEFAULT_VOICES["A"])
    ap.add_argument("--voice-b", default=tts_engine.DEFAULT_VOICES["B"])
    ap.add_argument("--rate", default="+0%", help="speech rate, e.g. +10%%")
    ap.add_argument("--script-only", action="store_true", help="write the transcript only, no audio")
    args = ap.parse_args(argv)

    title, text = read_source(args.input)
    title = args.title or title
    if args.max_chars and len(text) > args.max_chars:
        text = text[:args.max_chars]
    if len(text.strip()) < 100:
        print("Not enough text to make a podcast.", file=sys.stderr)
        return 2

    print(f"Source: {title}  ({len(text):,} chars)")
    base = os.path.splitext(args.out or args.input)[0]
    if base.endswith(".podcast"):
        base = base[: -len(".podcast")]
    episode = {"title": title, "author": args.author or _AUTHOR, "kind": args.kind, "mode": args.mode,
               "source_file": os.path.basename(args.input), "source_chars": len(text)}

    if args.mode == "llm":
        import llm
        if not llm.available():
            print("No LLM backend (Claude CLI or ANTHROPIC_API_KEY); falling back to rule mode.",
                  file=sys.stderr)
            args.mode = episode["mode"] = "rule"
    if args.mode == "llm":
        import grounding
        meta = {"title": title, "author": episode["author"] or "an unknown author", "kind": args.kind}
        ep = script_builder.build_llm(text, meta, minutes=args.minutes, check=not args.no_check,
                                      model=args.model)
        lines, sections, sents = ep["lines"], ep["sections"], ep["sents"]
        episode.update(minutes_target=args.minutes, checked=not args.no_check, sections=sections,
                       points=[p for p in ep["points"] if any(p["id"] in s["points"] for s in sections)],
                       n_points_mapped=len(ep["points"]))
        if not args.no_check:
            episode["check_rates"] = grounding.rates(lines)
            r = episode["check_rates"]
            print(f"  check: {r['claim_lines']} claim lines; first pass {r['first_pass']}; actions {r['actions']}")
    else:
        lines = script_builder.build_rule_based(text, title, sentences_per_segment=args.segment)
        sections, sents = None, None

    import textspans
    sents = sents or textspans.sentences(text)
    idx = textspans.index(sents)
    for ln in lines:
        ln["spans"] = [[idx[s]["start"], idx[s]["end"]] for s in ln.get("src", []) if s in idx]
    episode["lines"] = lines

    transcript = base + ".transcript.md"
    with open(transcript, "w", encoding="utf-8") as fh:
        fh.write(script_builder.script_to_markdown(lines, sections=sections, sents=sents))
    spoken = script_builder.spoken(lines)
    print(f"Transcript: {transcript}  ({len(spoken)} lines, {sum(len(l['text'].split()) for l in spoken):,} words)")

    if not args.script_only:
        out = args.out or (base + ".podcast.mp3")
        audio = tts_engine.synthesize(spoken, out, voices={"A": args.voice_a, "B": args.voice_b},
                                      rate=args.rate, sections=sections, title=title)
        for ln, tm in zip(spoken, audio.pop("timings")):
            ln["t"] = tm
        episode["audio"] = audio
        print(f"Done: {out}")

    with open(base + ".episode.json", "w", encoding="utf-8") as fh:
        json.dump(episode, fh, ensure_ascii=False, indent=1)
    print(f"Episode data: {base}.episode.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
