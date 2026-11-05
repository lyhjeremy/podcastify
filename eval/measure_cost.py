#!/usr/bin/env python3
"""Cost and wall time for one LLM-mode episode, end to end, with an empty cache.

The October eval did not record cost or time (and its disk cache means a rerun makes
no model calls). This makes one fresh 6-minute episode from the shortest eval source,
the Great Stink article (26,640 characters), exactly as `podcastify.py --mode llm`
would, and records every `claude -p` call: stage, duration_ms, total_cost_usd (the
API-equivalent cost; the run used a Claude subscription) and tokens. Stage wall
times include the edge-tts render. n = 1 episode.

    ~/venvs/podcastify/bin/python eval/measure_cost.py OUTDIR

OUTDIR gets the throwaway cache, transcript and audio (keep it outside the repo).
Writes eval/results/cost_calls.csv and eval/results/cost_episode.json.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.abspath(sys.argv[1])
os.makedirs(OUT, exist_ok=True)
os.environ["PODCASTIFY_CACHE"] = os.path.join(OUT, "cache")  # empty: every call is real

import grounding  # noqa: E402
import llm  # noqa: E402
import pipeline  # noqa: E402
import podcastify  # noqa: E402
import tts_engine  # noqa: E402

pipeline.CACHE_DIR = os.environ["PODCASTIFY_CACHE"]
SRC = os.path.expanduser("~/data-cache/podcastify/sources/greatstink.txt")
CALLS: list[dict] = []
STAGES: dict[str, float] = {}
_real_run = subprocess.run


def _stage_of_call() -> str:
    f = sys._getframe()
    while f:
        if f.f_code.co_name == "cached":
            return f.f_locals.get("tag", "?")
        f = f.f_back
    return "?"


def _spy_run(cmd, *a, **kw):
    t0 = time.perf_counter()
    out = _real_run(cmd, *a, **kw)
    wall = time.perf_counter() - t0
    if cmd and "claude" in os.path.basename(str(cmd[0])):
        try:
            d = json.loads(out.stdout)
        except Exception:
            d = {}
        u = d.get("usage") or {}
        CALLS.append({"stage": _stage_of_call(), "wall_s": round(wall, 2),
                      "duration_ms": d.get("duration_ms"), "cost_usd": d.get("total_cost_usd"),
                      "input_tokens": u.get("input_tokens"),
                      "cache_creation_input_tokens": u.get("cache_creation_input_tokens"),
                      "cache_read_input_tokens": u.get("cache_read_input_tokens"),
                      "output_tokens": u.get("output_tokens"),
                      "model": ",".join(sorted(d.get("modelUsage") or {}))})
        if len(CALLS) > 40:
            raise SystemExit("call cap of 40 reached")
    return out


llm.subprocess.run = _spy_run


def timed(mod, name, stage):
    fn = getattr(mod, name)

    def wrap(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            STAGES[stage] = STAGES.get(stage, 0) + time.perf_counter() - t0
    setattr(mod, name, wrap)


timed(pipeline, "map_pass", "map")
timed(pipeline, "reduce_pass", "reduce")
timed(pipeline, "script_pass", "script")
timed(grounding, "check_script", "check and repair")
timed(tts_engine, "synthesize", "audio (edge-tts)")

t0 = time.perf_counter()
podcastify.main([SRC, "--mode", "llm", "--minutes", "6", "--title", "Great Stink",
                 "--author", "Wikipedia contributors", "--kind", "encyclopedia article",
                 "-o", os.path.join(OUT, "greatstink.podcast.mp3")])
total = time.perf_counter() - t0

ep = json.load(open(os.path.join(OUT, "greatstink.episode.json")))
here = os.path.join(ROOT, "eval", "results")
fields = list(CALLS[0])
with open(os.path.join(here, "cost_calls.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(CALLS)
by_stage: dict[str, dict] = {}
for c in CALLS:
    s = by_stage.setdefault(c["stage"], {"calls": 0, "cost_usd": 0.0, "output_tokens": 0})
    s["calls"] += 1
    s["cost_usd"] = round(s["cost_usd"] + (c["cost_usd"] or 0), 5)
    s["output_tokens"] += c["output_tokens"] or 0
chip = _real_run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                 text=True).stdout.strip()
summary = {
    "date": date.today().isoformat(),
    "machine": f"{chip}, macOS {platform.mac_ver()[0]}",
    "source": "greatstink (26,640 chars), 6-minute target, empty cache, n = 1 episode",
    "cost_note": "total_cost_usd from claude -p --output-format json: the API-equivalent cost; "
                 "the run used a Claude subscription",
    "model_calls": len(CALLS),
    "cost_usd": round(sum(c["cost_usd"] or 0 for c in CALLS), 4),
    "wall_s_total": round(total, 1),
    "wall_s_by_stage": {k: round(v, 1) for k, v in STAGES.items()},
    "calls_by_stage": by_stage,
    "episode_minutes": round(ep["audio"]["duration"] / 60, 2) if ep.get("audio", {}).get("duration") else None,
    "spoken_words": sum(len(l["text"].split()) for l in ep["lines"] if not l.get("dropped")),
    "check_rates": ep.get("check_rates"),
}
with open(os.path.join(here, "cost_episode.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(json.dumps(summary, indent=1))
