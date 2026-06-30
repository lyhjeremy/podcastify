#!/usr/bin/env python3
"""
podcastify.py - turn an EPUB or text file into a two-host podcast.
==================================================================

    python podcastify.py book.epub                 # -> book.podcast.mp3 + transcript
    python podcastify.py article.txt --script-only # transcript only, no audio
    python podcastify.py book.epub --mode llm      # natural conversation (needs ANTHROPIC_API_KEY)
    python podcastify.py book.epub --voice-a en-US-GuyNeural --voice-b en-US-JennyNeural

Setup:
    pip install edge-tts ebooklib beautifulsoup4
    # ffmpeg on PATH for clean concatenation / .m4b

This tool ships no books or audio. Point it at a file you own or a public-domain
title; output is for personal use.
"""
from __future__ import annotations
import os
import sys
import argparse

import script_builder
import tts_engine


def read_source(path: str) -> tuple[str, str]:
    """Return (title, text) from an .epub or .txt/.md/.html file."""
    title = os.path.splitext(os.path.basename(path))[0]
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
        except Exception:
            pass
        chunks = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            chunks.append(soup.get_text(" ", strip=True))
        return title, "\n".join(chunks)
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
    ap.add_argument("-o", "--out", help="output audio path (default: <name>.podcast.mp3)")
    ap.add_argument("--mode", choices=["rule", "llm"], default="rule",
                    help="rule = two-voice reading with interjections; llm = Claude rewrite (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--max-chars", type=int, default=0,
                    help="cap source length (0 = whole file); handy for a quick preview")
    ap.add_argument("--segment", type=int, default=3, help="sentences per narration segment")
    ap.add_argument("--voice-a", default=tts_engine.DEFAULT_VOICES["A"])
    ap.add_argument("--voice-b", default=tts_engine.DEFAULT_VOICES["B"])
    ap.add_argument("--rate", default="+0%", help="speech rate, e.g. +10%")
    ap.add_argument("--script-only", action="store_true", help="write the transcript only, no audio")
    args = ap.parse_args(argv)

    title, text = read_source(args.input)
    if args.max_chars and len(text) > args.max_chars:
        text = text[:args.max_chars]
    if len(text.strip()) < 100:
        print("Not enough text to make a podcast.", file=sys.stderr)
        return 2

    print(f"Source: {title}  ({len(text):,} chars)")
    turns = script_builder.build_script(text, title=title, mode=args.mode,
                                        sentences_per_segment=args.segment) \
        if args.mode == "rule" else script_builder.build_script(text, title=title, mode=args.mode)

    base = os.path.splitext(args.out or args.input)[0]
    transcript = base + ".transcript.md"
    with open(transcript, "w", encoding="utf-8") as fh:
        fh.write(script_builder.script_to_markdown(turns))
    print(f"Transcript: {transcript}  ({len(turns)} turns)")

    if args.script_only:
        return 0

    out = args.out or (base + ".podcast.mp3")
    tts_engine.synthesize(
        turns, out,
        voices={"A": args.voice_a, "B": args.voice_b},
        rate=args.rate,
    )
    print(f"Done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
