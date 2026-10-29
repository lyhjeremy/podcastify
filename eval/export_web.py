#!/usr/bin/env python3
"""Write the files the write-up page reads.

    python eval/export_web.py [source]      # default source: alice

overview/episode.json  the recorded episode: every line with its timing, cited source
                       text and offsets, and both checkers' verdicts (dropped lines too)
overview/eval.json     the eval table, from eval/results/summary.json
overview/<source>.episode.mp3  re-encoded at 40 kbps from ~/data-cache/podcastify/eval-audio/<source>/llm.mp3
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import textspans  # noqa: E402

HERE = os.path.join(ROOT, "eval")
SRC = os.path.expanduser("~/data-cache/podcastify/sources")
AUDIO = os.path.expanduser("~/data-cache/podcastify/eval-audio")


def main(key: str = "alice") -> None:
    meta = json.load(open(os.path.join(HERE, "sources.json")))[key]
    summary = json.load(open(os.path.join(HERE, "results", "summary.json")))
    d = json.load(open(os.path.join(HERE, "results", f"{key}.lines.json")))
    text = open(os.path.join(SRC, f"{key}.txt"), encoding="utf-8").read()
    idx = textspans.index(textspans.sentences(text))
    s = summary[key]

    lines = []
    for ln in d["lines"]:
        c = ln["check"]
        ids = ln["src"] or ln.get("auto_src") or []
        lines.append({
            "who": ln["speaker"], "text": ln["text"], "sec": ln["section"],
            "t": ln.get("t"), "dropped": bool(ln.get("dropped")),
            "auto": not ln["src"],
            "spans": [[sid, idx[sid]["start"], idx[sid]["end"], text[idx[sid]["start"]:idx[sid]["end"]]]
                      for sid in ids if sid in idx],
            "first": c["first"], "why": c.get("why", ""), "action": c.get("action", "kept"),
            "final": c.get("final"), "after": c.get("after"), "was": c.get("original"),
            "sim": (c.get("local") or {}).get("sim"), "second": ln.get("second_check"),
        })
    ep = {"key": key, "title": meta["title"], "author": meta["author"], "kind": meta["kind"],
          "license": meta["license"], "url": meta["url"], "chars": s["chars"], "words": s["words_source"],
          "sentences": len(idx), "points_mapped": s["n_points_mapped"],
          "sections": [x["title"] for x in s["sections"]],
          "points_used": sum(len(x["points"]) for x in s["sections"]),
          "duration": s["conds"]["llm"].get("duration_s"), "chapters": s["conds"]["llm"].get("chapters"),
          "audio": f"{key}.episode.mp3", "lines": lines}
    out = os.path.join(ROOT, "overview")
    json.dump(ep, open(os.path.join(out, "episode.json"), "w", encoding="utf-8"), ensure_ascii=False)
    src_mp3 = os.path.join(AUDIO, key, "llm.mp3")
    if os.path.exists(src_mp3):
        # 40 kbps mono keeps the committed sample small; chapters carry over.
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src_mp3, "-map", "0:a", "-map_metadata", "0",
                        "-map_chapters", "0", "-c:a", "libmp3lame", "-b:a", "40k", "-ac", "1",
                        "-id3v2_version", "3", os.path.join(out, f"{key}.episode.mp3")], check=True)

    rows = []
    for k, r in summary.items():
        row = {"key": k, "title": r["title"], "kind": r["kind"], "license": r["license"],
               "chars": r["chars"], "pipeline": r["pipeline_check"], "listen": r["listen"], "conds": {}}
        for c, v in r["conds"].items():
            row["conds"][c] = {"coverage": v["coverage"]["score"], "of": v["coverage"]["of"],
                               "yes": v["coverage"]["yes"], "words": v["words"],
                               "duration_s": v.get("duration_s"), "wpm": v.get("wpm"),
                               "claim_lines": v["second_check"]["claim_lines"],
                               "partly": v["second_check"].get("partly"),
                               "unsupported": v["second_check"].get("unsupported"),
                               "not_supported_rate": v["second_check"]["not_supported_rate"]}
        rows.append(row)
    hc = os.path.join(HERE, "results", "hand_check_score.json")
    web = {"sources": rows, "hand_check": json.load(open(hc)) if os.path.exists(hc) else None}
    json.dump(web, open(os.path.join(out, "eval.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("wrote overview/episode.json, overview/eval.json")


if __name__ == "__main__":
    main(*sys.argv[1:])
