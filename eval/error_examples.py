#!/usr/bin/env python3
"""Pull the known failure cases out of the saved eval outputs into one file for the write-up.

    python eval/error_examples.py

Reads eval/results/*.lines.json, eval/results/summary.json and eval/keypoints/, plus the
map-pass replies in the LLM cache (~/data-cache/podcastify/llm-cache) to show which Darwin
key points the map step found and the reduce step left out. Makes no model calls.
Writes eval/results/error_examples.json.
"""
from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pipeline  # noqa: E402

RES = os.path.join(ROOT, "eval", "results")
CACHE = os.path.expanduser("~/data-cache/podcastify/llm-cache")
SOURCES = ["greatstink", "civil", "origin3", "alice"]


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as fh:
        return json.load(fh)


summary = load("summary.json")
lines = {s: load(f"{s}.lines.json") for s in SOURCES}
out: dict = {}

# 1. The end of Alice: both checkers read "this same little sister of hers" as the sister herself.
L = lines["alice"]["lines"]
end = []
for i in range(len(L) - 6, len(L) - 1):
    ln = L[i]
    c = ln["check"]
    end.append({"i": i, "speaker": ln["speaker"], "text": ln["text"], "src": ln.get("src"),
                "first": c.get("first"), "why": c.get("why"), "action": c.get("action"),
                "original": c.get("original"), "why_after": c.get("why_after"),
                "opus_on_draft": lines["alice"]["draft"][i].get("second_check")
                if i < len(lines["alice"]["draft"]) else None,
                "opus_on_checked": ln.get("second_check")})
out["alice_end"] = end

# 2. Darwin: key points the map step found, that the reduce step left out, against the
#    frozen key points Opus graded "no".
kp = {p["id"]: p["point"] for p in json.load(open(os.path.join(ROOT, "eval", "keypoints", "origin3.json")))["points"]}
marks = summary["origin3"]["conds"]["llm"]["coverage"]["marks"]
chosen = {p for sec in summary["origin3"]["sections"] for p in sec["points"]}
words = ["elephant", "357", "fulmar", "vermin", "clover", "scotch fir", "australia"]
mapped = []
for f in sorted(glob.glob(os.path.join(CACHE, "map-*.json"))):
    data = json.load(open(f, encoding="utf-8"))
    pts = data.get("points", []) if isinstance(data, dict) else data
    for p in pts:
        if any(w in p.get("point", "").lower() for w in words) and p["point"] not in [m["point"] for m in mapped]:
            mapped.append({"point": p["point"], "importance": p.get("importance"), "sents": p.get("sents")})
out["darwin"] = {"frozen_missing": {k: kp[k] for k, v in marks.items() if v == "no"},
                 "coverage_llm": summary["origin3"]["conds"]["llm"]["coverage"]["score"],
                 "outline_points": len(chosen),
                 "mapped_points": summary["origin3"]["n_points_mapped"],
                 "mapped_but_not_chosen": mapped}

# 3. Length: the script writer is asked for minutes * WPM words and writes more.
target_words = int(6 * pipeline.WPM)
out["length"] = {"target_minutes": 6, "target_words": target_words, "wpm_assumed": pipeline.WPM,
                 "sources": {s: {"draft_words": summary[s]["conds"]["llm-nocheck"]["words"],
                                 "checked_words": summary[s]["conds"]["llm"]["words"],
                                 "checked_minutes": round(summary[s]["conds"]["llm"]["duration_s"] / 60, 2),
                                 "checked_wpm": summary[s]["conds"]["llm"]["wpm"]}
                             for s in SOURCES}}

# 4. Questions tagged as supported claims.
qs = []
n_q = 0
opus_draft_q_supported = 0
opus_draft_q_flagged = []
for s in SOURCES:
    for i, ln in enumerate(lines[s]["lines"]):
        if not ln["text"].strip().endswith("?"):
            continue
        n_q += 1
        if ln["check"].get("first") in ("supported", "partly"):
            qs.append({"source": s, "text": ln["text"], "claim_flag": ln.get("claim"),
                       "src": ln.get("src"), "first": ln["check"].get("first"),
                       "why": ln["check"].get("why"), "opus": ln.get("second_check")})
    for ln in lines[s]["draft"]:
        if ln["text"].strip().endswith("?") and ln.get("second_check") in ("supported", "partly", "unsupported"):
            opus_draft_q_supported += 1
            if ln["second_check"] != "supported":
                opus_draft_q_flagged.append({"source": s, "text": ln["text"], "opus": ln["second_check"]})
out["questions"] = {"question_lines": n_q, "pipeline_rated": qs,
                    "opus_rated_as_claims_in_drafts": opus_draft_q_supported,
                    "opus_flagged_questions_in_drafts": opus_draft_q_flagged,
                    "opus_claim_lines_in_drafts": sum(summary[s]["conds"]["llm-nocheck"]["second_check"]["claim_lines"] for s in SOURCES)}

with open(os.path.join(RES, "error_examples.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print(json.dumps({k: (v if k != "alice_end" else len(v)) for k, v in out.items()}, indent=1, ensure_ascii=False)[:4000])
