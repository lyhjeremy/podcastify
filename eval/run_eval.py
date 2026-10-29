#!/usr/bin/env python3
"""Compare rule mode, LLM mode, and LLM mode without the grounding pass.

    PODCASTIFY_CACHE=~/data-cache/podcastify/llm-cache python eval/run_eval.py [--no-audio] [source ...]

For each source in eval/sources.json (fetched by fetch_sources.py):

  llm          the full pipeline: map, reduce, script, check, repair (Sonnet via claude -p)
  llm-nocheck  the same script before the check and repair step (the pipeline's draft)
  rule         rule mode, reading from the start until it has used as many words as llm

Measures, all written to eval/results/:
  coverage        Opus marks each frozen key point (eval/keypoints/, written before any
                  generation) as covered, partly covered or missing in each script.
  unsupported     Opus, as a second checker separate from the pipeline's Sonnet checker,
                  rates every llm and llm-nocheck line against its cited sentences. Rule mode
                  narration is the source verbatim, so it is supported by construction.
  listenability   Opus compares each pair of scripts blind, labelled only "Script 1" and
                  "Script 2", once in each order. A win needs both orders; otherwise a tie.
  duration, wpm   from rendering each script with edge-tts (skipped with --no-audio).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import grounding  # noqa: E402
import pipeline  # noqa: E402
import script_builder  # noqa: E402
import textspans  # noqa: E402
import tts_engine  # noqa: E402

HERE = os.path.join(ROOT, "eval")
SRC = os.path.expanduser("~/data-cache/podcastify/sources")
AUDIO = os.path.expanduser("~/data-cache/podcastify/eval-audio")
JUDGE = "opus"
MINUTES = 6
CONDS = ["rule", "llm-nocheck", "llm"]


def as_dialogue(lines):
    return "\n".join(f"{'Alex' if l['speaker'] == 'A' else 'Sam'}: {l['text']}" for l in lines)


COVER_SYSTEM = "You grade whether a podcast script covers a list of points. Reply with JSON only."
COVER_PROMPT = """Here is a list of key points about "{title}", and the script of a two-host audio episode about it.
For each point, say whether a listener of the episode would learn it:
"yes" (the script conveys the point, allowing paraphrase and minor omitted detail),
"partly" (it conveys some of the point but misses its substance), or "no".
Judge only what the script says, not whether it is accurate.

Reply as JSON: {{"points": [{{"id": "K1", "covered": "yes"}}]}}

KEY POINTS:
{points}

SCRIPT:
{script}
"""

LISTEN_SYSTEM = ("You are an experienced audio producer judging scripts for a two-host show. "
                 "Reply with JSON only.")
LISTEN_PROMPT = """Two scripts for an episode about "{title}" follow. Both will be read aloud by the same two
synthetic voices, Alex and Sam. Judge only listenability: which would be easier to follow and more
pleasant to listen to as audio, start to finish? Ignore which one is more complete or accurate.

Reply as JSON: {{"preferred": 1 or 2 or 0 for no preference, "why": "under 25 words"}}

SCRIPT 1:
{a}

SCRIPT 2:
{b}
"""


def coverage(title, keypoints, lines):
    pts = "\n".join(f"{k['id']}: {k['point']}" for k in keypoints)
    data = pipeline.cached("cover", COVER_PROMPT.format(title=title, points=pts, script=as_dialogue(lines)),
                           COVER_SYSTEM, JUDGE)
    marks = {p["id"]: p.get("covered", "no") for p in data.get("points", [])}
    score = sum(1.0 if marks.get(k["id"]) == "yes" else 0.5 if marks.get(k["id"]) == "partly" else 0.0
                for k in keypoints)
    return {"marks": marks, "score": score, "of": len(keypoints),
            "yes": sum(1 for v in marks.values() if v == "yes")}


def listen(title, scripts, a, b):
    out = []
    for first, second in ((a, b), (b, a)):
        data = pipeline.cached("listen", LISTEN_PROMPT.format(
            title=title, a=as_dialogue(scripts[first]), b=as_dialogue(scripts[second])), LISTEN_SYSTEM, JUDGE)
        p = int(data.get("preferred", 0) or 0)
        out.append({"order": [first, second], "preferred": first if p == 1 else second if p == 2 else None,
                    "why": data.get("why", "")})
    picks = [o["preferred"] for o in out]
    winner = picks[0] if picks[0] and picks[0] == picks[1] else None
    return {"pair": [a, b], "winner": winner, "orders": out}


def second_check(lines, text, sents, shown, meta):
    """Opus verdict on every line, against cited sentences (or the two nearest evidence sentences)."""
    idx = textspans.index(sents)
    items = []
    for n, ln in enumerate(lines):
        ids = ln["src"]
        if not ids:
            cands = [(s, text[idx[s]["start"]:idx[s]["end"]]) for s in shown.get(ln["section"], [])]
            ids = grounding.nearest(ln["text"], cands)
        items.append({"n": n + 1, "speaker": ln["speaker"], "text": ln["text"],
                      "prev": lines[n - 1]["text"] if n else "",
                      "cited": [(s, text[idx[s]["start"]:idx[s]["end"]]) for s in ids if s in idx]})
    v = grounding.llm_check(items, model=JUDGE, meta=meta)
    verdicts = [v.get(n + 1, {}).get("verdict", "unchecked") for n in range(len(lines))]
    claims = [x for x in verdicts if x in ("supported", "partly", "unsupported")]
    n = len(claims) or 1
    return {"verdicts": verdicts, "claim_lines": len(claims),
            "supported": claims.count("supported"), "partly": claims.count("partly"),
            "unsupported": claims.count("unsupported"),
            "unsupported_rate": round(claims.count("unsupported") / n, 3),
            "not_supported_rate": round((claims.count("unsupported") + claims.count("partly")) / n, 3)}


def words(lines):
    return sum(len(l["text"].split()) for l in lines)


def run_source(key, meta, audio=True):
    text = open(os.path.join(SRC, f"{key}.txt"), encoding="utf-8").read()
    kp = json.load(open(os.path.join(HERE, "keypoints", f"{key}.json")))["points"]
    print(f"== {key}: {meta['title']} ({len(text):,} chars)")
    ep = script_builder.build_llm(text, meta, minutes=MINUTES, check=True)
    final = script_builder.spoken(ep["lines"])
    draft = ep["draft"]
    rule = script_builder.build_rule_based(text, meta["title"], max_words=words(final), seed=1)
    scripts = {"rule": rule, "llm-nocheck": draft, "llm": final}

    res = {"source": key, "title": meta["title"], "kind": meta["kind"], "license": meta["license"],
           "chars": len(text), "words_source": len(text.split()),
           "n_points_mapped": len(ep["points"]), "sections": ep["sections"],
           "pipeline_check": grounding.rates(ep["lines"]), "conds": {}}
    for c, lines in scripts.items():
        res["conds"][c] = {"lines": len(lines), "words": words(lines),
                           "coverage": coverage(meta["title"], kp, lines)}
    for c in ("llm-nocheck", "llm"):
        res["conds"][c]["second_check"] = second_check(scripts[c], text, ep["sents"], ep["shown"], meta)
    res["conds"]["rule"]["second_check"] = {"note": "verbatim source; supported by construction",
                                            "claim_lines": sum(1 for l in rule if l["claim"]),
                                            "unsupported_rate": 0.0, "not_supported_rate": 0.0}
    res["listen"] = [listen(meta["title"], scripts, a, b)
                     for a, b in (("rule", "llm"), ("rule", "llm-nocheck"), ("llm-nocheck", "llm"))]
    if audio:
        os.makedirs(os.path.join(AUDIO, key), exist_ok=True)
        for c, lines in scripts.items():
            out = os.path.join(AUDIO, key, f"{c}.mp3")
            secs = ep["sections"] if c != "rule" else None
            a = tts_engine.synthesize(lines, out, sections=secs, title=meta["title"], progress=lambda *_: None)
            res["conds"][c]["duration_s"] = a["duration"]
            res["conds"][c]["wpm"] = round(words(lines) / (a["duration"] / 60), 1)
            res["conds"][c]["chapters"] = a["chapters"]
            if c == "llm":
                for ln, tm in zip(lines, a["timings"]):
                    ln["t"] = tm
            print(f"   {c:12s} {a['duration'] / 60:5.2f} min  {res['conds'][c]['wpm']} wpm")

    # Full line-level record for the llm condition (with checks), for the site and hand check.
    idx = textspans.index(ep["sents"])
    for ln in ep["lines"]:
        ln["spans"] = [[idx[s]["start"], idx[s]["end"]] for s in ln["src"] if s in idx]
        if ln.get("auto_src"):
            ln["auto_spans"] = [[idx[s]["start"], idx[s]["end"]] for s in ln["auto_src"] if s in idx]
    for ln, v in zip(final, res["conds"]["llm"]["second_check"]["verdicts"]):
        ln["second_check"] = v
    for ln, v in zip(draft, res["conds"]["llm-nocheck"]["second_check"]["verdicts"]):
        ln["second_check"] = v
    detail = {"source": key, "lines": ep["lines"], "draft": draft,
              "points": [p for p in ep["points"] if any(p["id"] in s["points"] for s in ep["sections"])]}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(detail, open(os.path.join(HERE, "results", f"{key}.lines.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=0)
    for c in CONDS:
        r = res["conds"][c]
        print(f"   {c:12s} coverage {r['coverage']['score']:4.1f}/{r['coverage']['of']}  "
              f"not supported {r['second_check']['not_supported_rate']:.1%}  words {r['words']}")
    print("   listen:", [(l["pair"], l["winner"]) for l in res["listen"]])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="*")
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()
    meta = json.load(open(os.path.join(HERE, "sources.json")))
    keys = args.sources or list(meta)
    out_path = os.path.join(HERE, "results", "summary.json")
    summary = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for k in keys:
        summary[k] = run_source(k, meta[k], audio=not args.no_audio)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(summary, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    random.seed(0)


if __name__ == "__main__":
    main()
