#!/usr/bin/env python3
"""A person-checkable sample of the grounding checker.

    python eval/hand_check.py sample   # writes eval/results/hand_sample.md, verdicts hidden
    python eval/hand_check.py score    # compares eval/hand_check.json with the checkers

The sample is 20 claim lines drawn at random (seed 19) from the LLM drafts of all
eval sources, before any repair, so failing lines can turn up. Each comes with the
source sentences it cites. A person reads each line against those sentences, records
supported / partly / unsupported in eval/hand_check.json, and only then runs `score`.
"""
from __future__ import annotations

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SRC = os.path.expanduser("~/data-cache/podcastify/sources")


def pool():
    out = []
    for key in sorted(json.load(open(os.path.join(HERE, "sources.json")))):
        path = os.path.join(RES, f"{key}.lines.json")
        if not os.path.exists(path):
            continue
        text = open(os.path.join(SRC, f"{key}.txt"), encoding="utf-8").read()
        d = json.load(open(path))
        for i, ln in enumerate(d["lines"]):
            c = ln["check"]
            if c["first"] not in ("supported", "partly", "unsupported"):
                continue
            spans = ln.get("spans") or ln.get("auto_spans") or []
            draft_text = c.get("original", ln["text"])
            # The draft's own citations: for rewritten lines the repair may have changed them.
            dsrc = d["draft"][i]["src"] if i < len(d["draft"]) else ln["src"]
            out.append({"id": f"{key}:{i + 1}", "text": draft_text, "src": dsrc,
                        "sources_text": [text[a:b] for a, b in spans] if dsrc == ln["src"] else None,
                        "pipeline": c["first"], "second": d["draft"][i].get("second_check")
                        if i < len(d["draft"]) else None, "key": key, "i": i})
    return out


def cited_text(item):
    if item["sources_text"] is not None:
        return item["sources_text"]
    # Rewritten line with new citations: rebuild the draft's own cited sentences.
    sys.path.insert(0, os.path.dirname(HERE))
    import textspans
    text = open(os.path.join(SRC, f"{item['key']}.txt"), encoding="utf-8").read()
    idx = textspans.index(textspans.sentences(text))
    return [text[idx[s]["start"]:idx[s]["end"]] for s in item["src"] if s in idx]


def sample():
    rng = random.Random(19)
    items = rng.sample(pool(), 20)
    lines = ["# Hand check sample (verdicts hidden)\n"]
    for it in items:
        lines.append(f"## {it['id']}\n\nLine: {it['text']}\n\nCited ({', '.join(it['src']) or 'auto'}):\n")
        lines += [f"> {t}\n" for t in cited_text(it)]
    open(os.path.join(RES, "hand_sample.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", os.path.join(RES, "hand_sample.md"), [i["id"] for i in items])


def score():
    mine = json.load(open(os.path.join(HERE, "hand_check.json")))["verdicts"]
    by_id = {p["id"]: p for p in pool()}
    rows, agree_p, agree_s, agree_bin = [], 0, 0, 0
    for vid, v in mine.items():
        p = by_id[vid]
        rows.append({"id": vid, "hand": v["verdict"], "pipeline": p["pipeline"], "second": p["second"],
                     "note": v.get("note", "")})
        agree_p += v["verdict"] == p["pipeline"]
        agree_s += v["verdict"] == p["second"]
        agree_bin += (v["verdict"] == "supported") == (p["pipeline"] == "supported")
    n = len(rows)
    out = {"n": n, "agree_pipeline_exact": agree_p, "agree_second_exact": agree_s,
           "agree_pipeline_pass_fail": agree_bin, "rows": rows}
    json.dump(out, open(os.path.join(RES, "hand_check_score.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}))


if __name__ == "__main__":
    {"sample": sample, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "sample"]()
