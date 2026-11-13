#!/usr/bin/env python3
"""Run every checker over the frozen judge set (eval/judges/judge_set.json).

    python eval/judges/run_judges.py local                  # MiniLM similarity and number check
    python eval/judges/run_judges.py llm --model haiku --run 1
    python eval/judges/run_judges.py llm --model sonnet --run 1
    python eval/judges/run_judges.py llm --model opus --run 1
    python eval/judges/run_judges.py reparse                # re-read saved replies, no calls

The LLM judges get exactly the pipeline's check: grounding.CHECK_SYSTEM and
grounding.CHECK_PROMPT, one batch per source of at most 18 lines, the previous line
for context, the cited sentences under each line. An original and its perturbed copy
are always put in different batches, so a judge never sees both side by side.
Calls bypass the pipeline's prompt cache (each run is a fresh call) and record the
CLI's total_cost_usd and duration_ms. Outputs: eval/judges/outputs/<judge>_r<run>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import grounding  # noqa: E402
import llm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
BATCH = 18
SEED = 11


def load_set():
    with open(os.path.join(HERE, "judge_set.json"), encoding="utf-8") as fh:
        return json.load(fh)["items"]


def load_meta():
    with open(os.path.join(ROOT, "eval", "sources.json"), encoding="utf-8") as fh:
        return json.load(fh)


def batches(items):
    """Per source, two or more batches; pair members go to different batches."""
    rng = random.Random(SEED)
    out = []
    for source in dict.fromkeys(it["source"] for it in items):
        mine = [it for it in items if it["source"] == source]
        k = max(2, -(-len(mine) // BATCH))
        groups = [[] for _ in range(k)]
        pair_slot, spare = {}, 0
        for it in mine:
            if it.get("pair"):
                if it["pair"] not in pair_slot:
                    pair_slot[it["pair"]] = len(pair_slot) % k
                    g = pair_slot[it["pair"]]
                else:
                    g = (pair_slot[it["pair"]] + 1) % k
            else:
                g = spare % k
                spare += 1
            groups[g].append(it)
        for g in groups:
            rng.shuffle(g)
            out.append((source, g))
    return out


def call_cli(prompt, system, model, timeout=600):
    cmd = [llm._which_claude() or "claude", "-p", "--tools", "", "--setting-sources", "",
           "--no-session-persistence", "--output-format", "json", "--system-prompt", system, "--model", model]
    with tempfile.TemporaryDirectory() as cwd:
        t0 = time.time()
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        wall = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[:300])
    env = json.loads(proc.stdout)
    return env, wall


def extract(raw, ids):
    """{item id: {"verdict", "why"}} from a reply; lines without a verdict are left out."""
    try:
        data = llm.parse_json(raw)
    except Exception:  # scored as missing verdicts
        data = {}
    verdicts = {}
    for v in (data.get("verdicts", []) if isinstance(data, dict) else []):
        try:
            n = int(v.get("n"))
        except (TypeError, ValueError):
            continue
        if 1 <= n <= len(ids):
            verdicts[ids[n - 1]] = {"verdict": str(v.get("verdict", "")).lower().strip(), "why": v.get("why", "")}
    return verdicts


def reparse():
    """Re-read every saved reply with the current llm.parse_json (no model calls). Records how
    many verdicts the parser at run time had found, so the effect of a parser fix is visible."""
    for path in sorted(os.listdir(OUT)):
        if not path.endswith(".json") or path == "local.json":
            continue
        full = os.path.join(OUT, path)
        d = json.load(open(full, encoding="utf-8"))
        for b in d["batches"]:
            b.setdefault("verdicts_at_run_time", len(b["verdicts"]))
            b["verdicts"] = extract(b["raw"], b["ids"])
        with open(full, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=1)
        print(path, sum(b["verdicts_at_run_time"] for b in d["batches"]), "->", sum(len(b["verdicts"]) for b in d["batches"]))


def run_llm(model, run):
    items, meta = load_set(), load_meta()
    dest = os.path.join(OUT, f"{model}_r{run}.json")
    done = json.load(open(dest, encoding="utf-8")) if os.path.exists(dest) else {
        "judge": model, "run": run, "prompt": "grounding.CHECK_PROMPT", "batches": []}
    finished = {tuple(b["ids"]) for b in done["batches"]}
    plan = batches(items)
    for bi, (source, group) in enumerate(plan):
        ids = [it["id"] for it in group]
        if tuple(ids) in finished:
            continue
        rendered = [{"n": n + 1, "speaker": it["speaker"], "text": it["text"], "prev": it["prev"],
                     "cited": [tuple(c) for c in it["cited"]]} for n, it in enumerate(group)]
        m = meta[source]
        prompt = grounding.CHECK_PROMPT.format(items=grounding._render(rendered), title=m["title"], author=m["author"])
        env, wall = call_cli(prompt, grounding.CHECK_SYSTEM, model)
        verdicts = extract(env.get("result", ""), ids)
        done["batches"].append({"source": source, "ids": ids, "verdicts": verdicts,
                                "cost_usd": env.get("total_cost_usd"), "duration_ms": env.get("duration_ms"),
                                "wall_s": round(wall, 2), "usage": env.get("usage"), "raw": env.get("result", "")})
        os.makedirs(OUT, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(done, fh, ensure_ascii=False, indent=1)
        print(f"{model} r{run} batch {bi + 1}/{len(plan)} {source}: {len(verdicts)}/{len(ids)} verdicts "
              f"${env.get('total_cost_usd', 0):.4f} {env.get('duration_ms')} ms", flush=True)


def run_local():
    items = load_set()
    out = {"judge": "local", "model": grounding.LOCAL_MODEL, "threshold": grounding.LOCAL_THRESHOLD, "items": {}}
    t0 = time.perf_counter()
    grounding.local_check("warm up", ["warm up the encoder"])
    warm = time.perf_counter() - t0
    times = []
    for it in items:
        t = time.perf_counter()
        r = grounding.local_check(it["text"], [c[1] for c in it["cited"]])
        times.append(time.perf_counter() - t)
        out["items"][it["id"]] = {"sim": r["sim"], "numbers_missing": r["numbers_missing"],
                                  "minilm_flag": r["sim"] is not None and r["sim"] < grounding.LOCAL_THRESHOLD,
                                  "numbers_flag": bool(r["numbers_missing"])}
    times.sort()
    out["timing"] = {"machine": "Apple M4", "model_load_s": round(warm, 2),
                     "median_ms": round(1000 * times[len(times) // 2], 1),
                     "p95_ms": round(1000 * times[int(0.95 * (len(times) - 1))], 1)}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "local.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("local:", out["timing"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["local", "llm", "plan", "reparse"])
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--run", type=int, default=1)
    a = ap.parse_args()
    if a.what == "local":
        run_local()
    elif a.what == "reparse":
        reparse()
    elif a.what == "plan":
        for s, g in batches(load_set()):
            print(s, len(g), sorted(it["kind"] for it in g).count("perturbed"))
    else:
        run_llm(a.model, a.run)
