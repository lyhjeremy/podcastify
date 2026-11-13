#!/usr/bin/env python3
"""Score every checker on the frozen judge set and write the web data.

    python eval/judges/score.py   # -> eval/judges/results.json, overview/judges.json

Positive class: "not fully supported" (a judge says partly or unsupported; MiniLM
similarity under 0.40; a digit in the line that the cited text lacks). Precision and
recall are measured on the 112 labeled items (56 originals presumed supported, 56
perturbed copies not supported by construction). Agreement (Cohen's kappa) is over
all 124 items, flagged lines included.
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import grounding  # noqa: E402
import textspans  # noqa: E402

SRC = os.path.expanduser("~/data-cache/podcastify/sources")


def context(items):
    """The sentence before and after each item's cited sentences, for the web viewer."""
    out, cache = {}, {}
    for it in items:
        if it["source"] not in cache:
            text = open(os.path.join(SRC, f"{it['source']}.txt"), encoding="utf-8").read()
            sents = textspans.sentences(text)
            cache[it["source"]] = (text, sents, {s["id"]: k for k, s in enumerate(sents)})
        text, sents, pos = cache[it["source"]]
        ks = sorted(pos[s] for s, _ in it["cited"] if s in pos)
        if not ks:
            out[it["id"]] = ["", ""]
            continue
        before = sents[ks[0] - 1] if ks[0] > 0 else None
        after = sents[ks[-1] + 1] if ks[-1] + 1 < len(sents) else None
        out[it["id"]] = [text[before["start"]:before["end"]] if before else "",
                         text[after["start"]:after["end"]] if after else ""]
    return out

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
POS = ("partly", "unsupported")
LLM_JUDGES = ["haiku", "sonnet", "opus"]


def load(name):
    with open(name, encoding="utf-8") as fh:
        return json.load(fh)


def kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def pr(flags: dict, items: list[dict]) -> dict:
    lab = [it for it in items if it["label"]]
    tp = sum(1 for it in lab if it["label"] == "not_supported" and flags[it["id"]])
    fn = sum(1 for it in lab if it["label"] == "not_supported" and not flags[it["id"]])
    fp = sum(1 for it in lab if it["label"] == "supported" and flags[it["id"]])
    tn = sum(1 for it in lab if it["label"] == "supported" and not flags[it["id"]])
    by_type = {}
    for t in ("number", "name", "negation"):
        sel = [it for it in lab if it.get("perturbation") == t]
        by_type[t] = {"n": len(sel), "caught": sum(1 for it in sel if flags[it["id"]])}
    rec = tp / (tp + fn) if tp + fn else None
    prec = tp / (tp + fp) if tp + fp else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    flagged_pool = [it for it in items if it["kind"] == "flagged"]
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": round(100 * rec, 1) if rec is not None else None,
            "precision": round(100 * prec, 1) if prec is not None else None,
            "f1": round(100 * f1, 1) if f1 is not None else None,
            "false_flag_rate": round(100 * fp / (fp + tn), 1) if fp + tn else None,
            "by_perturbation": by_type,
            "flagged_pool_flags": sum(1 for it in flagged_pool if flags[it["id"]])}


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))] if xs else None


def question_recount():
    """Published second-check rates, recounted with questions taken out of the claim lines."""
    out = {}
    for cond in ("draft", "lines"):
        claims = flagged = q_claims = q_flagged = 0
        for path in sorted(glob.glob(os.path.join(ROOT, "eval", "results", "*.lines.json"))):
            data = load(path)
            for ln in data[cond]:
                v = ln.get("second_check")
                if v not in ("supported", "partly", "unsupported"):
                    continue
                if ln.get("dropped"):
                    continue
                if grounding.is_question(ln["text"]):
                    q_claims += 1
                    q_flagged += v in POS
                else:
                    claims += 1
                    flagged += v in POS
        out["draft" if cond == "draft" else "final"] = {
            "published_claim_lines": claims + q_claims, "published_not_supported": flagged + q_flagged,
            "question_lines": q_claims, "questions_not_supported": q_flagged,
            "statement_claim_lines": claims, "statement_not_supported": flagged}
    return out


def main():
    items = load(os.path.join(HERE, "judge_set.json"))["items"]
    ids = [it["id"] for it in items]
    local = load(os.path.join(OUT, "local.json"))
    runs = {}
    for path in sorted(glob.glob(os.path.join(OUT, "*_r*.json"))):
        d = load(path)
        runs[f"{d['judge']}_r{d['run']}"] = d

    verdict = {}  # judge_run -> id -> verdict
    for key, d in runs.items():
        v = {}
        for b in d["batches"]:
            for i in b["ids"]:
                v[i] = b["verdicts"].get(i, {"verdict": "missing", "why": ""})
        verdict[key] = v

    flags = {
        "minilm": {i: local["items"][i]["minilm_flag"] for i in ids},
        "numbers": {i: local["items"][i]["numbers_flag"] for i in ids},
    }
    flags["local"] = {i: flags["minilm"][i] or flags["numbers"][i] for i in ids}
    for key, v in verdict.items():
        flags[key] = {i: v[i]["verdict"] in POS for i in ids}
    r1 = [f"{j}_r1" for j in LLM_JUDGES if f"{j}_r1" in flags]
    flags["majority_llm"] = {i: sum(flags[k][i] for k in r1) >= 2 for i in ids}
    five = ["minilm", "numbers"] + r1
    flags["majority_all5"] = {i: sum(flags[k][i] for k in five) >= 3 for i in ids}

    judges = {}
    for key in ["minilm", "numbers", "local"] + sorted(verdict) + ["majority_llm", "majority_all5"]:
        judges[key] = pr(flags[key], items)
        if key in verdict:
            d = runs[key]
            cost = sum(b.get("cost_usd") or 0 for b in d["batches"])
            walls = [b["wall_s"] for b in d["batches"]]
            n_lines = sum(len(b["ids"]) for b in d["batches"])
            judges[key].update({
                "calls": len(d["batches"]), "lines": n_lines, "missing_verdicts": sum(1 for i in ids if verdict[key][i]["verdict"] == "missing"),
                "total_cost_usd": round(cost, 4), "cost_per_line_usd": round(cost / n_lines, 5),
                "cost_per_1000_lines_usd": round(1000 * cost / n_lines, 2),
                "batch_latency_median_s": round(statistics.median(walls), 1), "batch_latency_p95_s": round(p95(walls), 1),
                "latency_per_line_s": round(sum(walls) / n_lines, 2),
                "verdict_counts": {x: sum(1 for i in ids if verdict[key][i]["verdict"] == x)
                                   for x in ("supported", "partly", "unsupported", "no_claim", "missing")}})

    # Run-to-run stability per LLM judge.
    stability = {}
    for j in LLM_JUDGES:
        keys = sorted(k for k in verdict if k.startswith(j + "_r"))
        if len(keys) < 2:
            continue
        same_bin = sum(1 for i in ids if len({flags[k][i] for k in keys}) == 1)
        same_verdict = sum(1 for i in ids if len({verdict[k][i]["verdict"] for k in keys}) == 1)
        recs = [judges[k]["recall"] for k in keys]
        ffr = [judges[k]["false_flag_rate"] for k in keys]
        stability[j] = {"runs": len(keys), "n": len(ids),
                        "same_flag_all_runs": same_bin, "same_flag_pct": round(100 * same_bin / len(ids), 1),
                        "same_verdict_all_runs": same_verdict,
                        "kappa_r1_r2": kappa([flags[keys[0]][i] for i in ids], [flags[keys[1]][i] for i in ids]),
                        "recall_by_run": recs, "false_flag_rate_by_run": ffr}

    order = ["minilm", "numbers"] + r1 + ["majority_llm"]
    agreement = {f"{a}|{b}": kappa([flags[a][i] for i in ids], [flags[b][i] for i in ids])
                 for x, a in enumerate(order) for b in order[x + 1:]}

    res = {"n_items": len(items), "n_labeled": sum(1 for it in items if it["label"]),
           "counts": load(os.path.join(HERE, "judge_set.json"))["counts"],
           "positive_class": "partly or unsupported (LLM); sim < 0.40 (MiniLM); digit missing from cited text (numbers)",
           "majority_llm": f"2 of 3 of {r1}", "majority_all5": f"3 of 5 of {five}",
           "cost_note": "API-equivalent cost reported by the Claude CLI (total_cost_usd); the CLI runs on a subscription.",
           "local_timing": local["timing"], "judges": judges, "stability": stability, "kappa": agreement,
           "question_recount": question_recount()}

    # Error analysis: majority misses and false flags, with each judge's reason.
    def row(it):
        return {"id": it["id"], "source": it["source"], "kind": it["kind"], "edit": it.get("edit"),
                "text": it["text"], "verdicts": {k: verdict[k][it["id"]]["verdict"] for k in r1},
                "why": {k: verdict[k][it["id"]]["why"] for k in r1}, "sim": local["items"][it["id"]]["sim"]}
    res["errors"] = {
        "perturbed_missed_by_majority": [row(it) for it in items if it["kind"] == "perturbed" and not flags["majority_llm"][it["id"]]],
        "originals_flagged_by_majority": [row(it) for it in items if it["kind"] == "original" and flags["majority_llm"][it["id"]]],
    }
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)

    # Web data for the span viewer.
    ctx = context(items)
    web = {"judges": order, "runs": {j: sorted(k for k in verdict if k.startswith(j + "_r")) for j in LLM_JUDGES},
           "threshold": grounding.LOCAL_THRESHOLD, "items": []}
    for it in items:
        web["items"].append({
            "id": it["id"], "source": it["source"], "kind": it["kind"], "label": it["label"],
            "perturbation": it.get("perturbation"), "edit": it.get("edit"), "original_text": it.get("original_text"),
            "speaker": it["speaker"], "prev": it["prev"], "text": it["text"], "cited": it["cited"],
            "before": ctx[it["id"]][0], "after": ctx[it["id"]][1],
            "sim": local["items"][it["id"]]["sim"], "numbers_missing": local["items"][it["id"]]["numbers_missing"],
            "verdicts": {k: verdict[k][it["id"]] for k in sorted(verdict)},
            "majority": flags["majority_llm"][it["id"]]})
    with open(os.path.join(ROOT, "overview", "judges.json"), "w", encoding="utf-8") as fh:
        json.dump(web, fh, ensure_ascii=False, separators=(",", ":"))
    summary = {k: (v["recall"], v["precision"], v["false_flag_rate"]) for k, v in judges.items()}
    print(json.dumps(summary, indent=0))
    print(json.dumps(stability, indent=0))
    print(json.dumps(agreement, indent=0))
    print(json.dumps(res["question_recount"], indent=0))


if __name__ == "__main__":
    main()
