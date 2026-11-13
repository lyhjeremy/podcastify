#!/usr/bin/env python3
"""Build the frozen judge-eval set: script lines with a known answer to "is this
line fully supported by the sentences it cites?". No model call is made here.

    python eval/judges/build_set.py        # -> eval/judges/judge_set.json

Three kinds of item, all from the saved scripts in eval/results/<source>.lines.json
(drafts and final episodes, duplicates removed, cited sentences read back from the
source text):

  original   a statement line, drawn at random (seed 7) from all lines that cite at
             least one sentence, whatever the earlier checkers said about it. Label "supported" is presumed, not known: the
             published eval had Opus rate about 5% of claim lines partly supported,
             so a few originals may be wrong. Questions are left out (see below).
  perturbed  the same line with one edit made by code, so the label is known by
             construction: the edited line says something its cited sentences do not.
               number    a figure in digits is changed (a year moves by 10, other
                         numbers by a fixed step) and the new figure must not occur in
                         the cited text
               name      a person, place or character named in the line is swapped for
                         another one from the same source that the cited text does not
                         mention
               negation  the first auxiliary verb is negated ("was" -> "was not") or an
                         existing negation is removed ("didn't" -> "did"), and the result
                         must not occur in the cited text
  flagged    every other statement line that any earlier checker (pipeline Sonnet, eval Opus,
             MiniLM or the number check) rated as not fully supported. No label: these
             are the disputed lines, scored for agreement only and written to the
             labeling sheet.

Questions (every sentence of the line ends in "?") are excluded from the judge set and
handled by grounding.is_question, which keeps them out of the claim-line rates.
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import grounding  # noqa: E402
import textspans  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expanduser("~/data-cache/podcastify/sources")
SOURCES = ["greatstink", "civil", "origin3", "alice"]
SEED = 7
PER_SOURCE = 14  # originals drawn per source, each gets one perturbation if one applies

# Names a line may mention, per source. A swap takes another name from the same list.
NAMES = {
    "greatstink": {
        "person": ["Bazalgette", "John Snow", "Snow", "Edwin Chadwick", "Chadwick", "Michael Faraday", "Faraday",
                   "Disraeli", "Benjamin Disraeli", "William Farr", "Farr", "Goldsworthy Gurney", "Gurney",
                   "Peter Ackroyd", "Ackroyd", "John Doxat", "Doxat", "Queen Victoria", "Prince Albert"],
        "place": ["Westminster", "Lambeth", "Southwark", "Crossness", "Abbey Mills", "Chelsea", "Thames", "London"],
        "body": ["Punch", "The Times", "Metropolitan Board of Works", "Parliament"]},
    "civil": {
        "person": ["Thoreau", "Washington", "Franklin", "Webster", "Daniel Webster", "Confucius", "Christ",
                   "Luther", "Copernicus", "Paley", "Caesar", "Orpheus"],
        "place": ["Massachusetts", "Mexico", "Concord", "Virginia", "Carolina", "England"]},
    "origin3": {
        "person": ["Darwin", "Malthus", "De Candolle", "Linnaeus", "Lyell", "Herbert", "Wallace"],
        "place": ["La Plata", "Australia", "England", "Scotland", "Staffordshire", "Paraguay", "India", "Europe",
                  "America", "Surrey", "Farnham"]},
    "alice": {
        "character": ["Alice", "White Rabbit", "Rabbit", "Hatter", "March Hare", "Dormouse", "Queen", "King",
                      "Duchess", "Cheshire Cat", "Cat", "Caterpillar", "Mock Turtle", "Gryphon", "Dodo", "Mouse",
                      "Bill", "Knave", "Dinah", "Pigeon", "Mabel"]},
}

NEGATE = [(r"\bwas\b", "was not"), (r"\bwere\b", "were not"), (r"\bis\b", "is not"), (r"\bare\b", "are not"),
          (r"\bcan\b", "cannot"), (r"\bwill\b", "will not"), (r"\bwould\b", "would not"), (r"\bhad\b", "had not"),
          (r"\bcould\b", "could not"), (r"\bshould\b", "should not"), (r"\bmust\b", "must not")]
UNNEGATE = [(r"\bdidn't\b", "did"), (r"\bdoesn't\b", "does"), (r"\bdon't\b", "do"), (r"\bisn't\b", "is"),
            (r"\bwasn't\b", "was"), (r"\baren't\b", "are"), (r"\bcan't\b", "can"), (r"\bcannot\b", "can"),
            (r"\bwon't\b", "will"), (r"\bnever\b", "always"), (r"\bnot\b", "")]


def is_question(text: str) -> bool:
    return grounding.is_question(text)


def perturb_number(line: str, cited: str):
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", line):
        raw = m.group(0)
        n = raw.replace(",", "")
        if "." in n:
            continue
        v = int(n)
        new = v + 10 if 1000 <= v <= 2100 else (v + 1 if v < 10 else int(round(v * 1.5)))
        new_s = f"{new:,}" if "," in raw else str(new)
        if new_s in cited or str(new) in cited.replace(",", ""):
            continue
        return line[:m.start()] + new_s + line[m.end():], f"{raw} -> {new_s}"
    return None


def perturb_name(line: str, cited: str, groups: dict[str, list[str]], rng: random.Random):
    """Swap a name for another of the same kind (person for person, place for place)."""
    found = [(len(n), n, kind) for kind, names in groups.items() for n in names
             if re.search(rf"\b{re.escape(n)}\b", line)]
    for _, name, kind in sorted(found, reverse=True):
        m = re.search(rf"\b{re.escape(name)}\b", line)
        options = [x for x in groups[kind] if x not in line and x not in cited and name not in x and x not in name]
        if not options:
            continue
        new = rng.choice(sorted(options))
        return line[:m.start()] + new + line[m.end():], f"{name} -> {new}"
    return None


def perturb_negation(line: str, cited: str):
    """Negate the first auxiliary, or remove an existing negation. Skipped where the
    result would be a double negative ("is not no") or a bare sentence-initial "Not"."""
    norm = line.replace("\u2019", "'")
    for pat, rep in UNNEGATE + NEGATE:
        for m in re.finditer(pat, norm, flags=re.I):
            word = m.group(0)
            after = norm[m.end():].lstrip().lower()
            before = norm[:m.start()].rstrip()
            if after.startswith(("no ", "not ", "never ", "nothing")):
                continue
            if word.lower() == "not" and (not before or before[-1] in ".!?\""):
                continue
            new_word = rep if word[0].islower() or not rep else rep[0].upper() + rep[1:]
            out = re.sub(r"\s{2,}", " ", norm[:m.start()] + new_word + norm[m.end():]).strip()
            out = out.replace(" ,", ",")
            if new_word and f" {new_word.lower()} " in f" {cited.lower().replace(chr(8217), chr(39))} ":
                continue
            return out, f"{word} -> {new_word or '(removed)'}"
    return None


def collect(source: str):
    text = open(os.path.join(SRC, f"{source}.txt"), encoding="utf-8").read()
    sents = textspans.sentences(text)
    idx = textspans.index(sents)
    data = json.load(open(os.path.join(ROOT, "eval", "results", f"{source}.lines.json"), encoding="utf-8"))
    final_by_text = {ln["text"]: ln for ln in data["lines"]}
    seen, rows = set(), []
    for cond in ("draft", "lines"):
        script = data[cond]
        for n, ln in enumerate(script):
            src = [s for s in ln.get("src", []) if s in idx]
            # n == 0 is the greeting that names the work and author: never a claim.
            if not src or n == 0 or ln["text"] in seen:
                continue
            seen.add(ln["text"])
            fin = final_by_text.get(ln["text"], {})
            chk = fin.get("check") or ln.get("check") or {}
            rows.append({
                "source": source, "script": "draft" if cond == "draft" else "final", "pos": n,
                "speaker": ln["speaker"], "text": ln["text"], "prev": script[n - 1]["text"] if n else "",
                "src": src, "cited": [[s, text[idx[s]["start"]:idx[s]["end"]]] for s in src],
                "spans": [[idx[s]["start"], idx[s]["end"]] for s in src],
                "earlier": {"opus": ln.get("second_check"), "sonnet": chk.get("first"),
                            "local_flag": (chk.get("local") or {}).get("flag")},
            })
    return rows


def flagged(r: dict) -> bool:
    e = r["earlier"]
    return e["opus"] in ("partly", "unsupported") or e["sonnet"] in ("partly", "unsupported") or bool(e["local_flag"])


def main():
    rng = random.Random(SEED)
    items, questions_excluded = [], 0
    for source in SOURCES:
        rows = collect(source)
        stmts = []
        for r in rows:
            if is_question(r["text"]):
                questions_excluded += 1
                continue
            stmts.append(r)
        pool = list(stmts)
        rng.shuffle(pool)
        chosen = set()
        # Prefer lines where some perturbation applies; numbers first (rarest), then names, then negation.
        picked = 0
        for r in pool:
            if picked >= PER_SOURCE:
                break
            cited = " ".join(t for _, t in r["cited"])
            for kind, fn in (("number", lambda: perturb_number(r["text"], cited)),
                             ("name", lambda: perturb_name(r["text"], cited, NAMES[source], rng)),
                             ("negation", lambda: perturb_negation(r["text"], cited))):
                got = fn()
                if got:
                    break
            if not got:
                continue
            new_text, edit = got
            pid = f"{source}-{r['script']}-{r['pos']}"
            items.append({**r, "kind": "original", "label": "supported", "pair": pid})
            items.append({**r, "text": new_text, "kind": "perturbed", "label": "not_supported", "pair": pid,
                          "perturbation": kind, "edit": edit, "original_text": r["text"]})
            picked += 1
            chosen.add(r["text"])
        # Flagged lines not already drawn as originals (an original keeps its earlier verdicts).
        for r in stmts:
            if flagged(r) and r["text"] not in chosen:
                items.append({**r, "kind": "flagged", "label": None})
    for k, it in enumerate(items):
        it["id"] = f"J{k + 1:03d}"
    counts = {}
    for it in items:
        key = it["kind"] if it["kind"] != "perturbed" else f"perturbed:{it['perturbation']}"
        counts[key] = counts.get(key, 0) + 1
    out = {"built_by": "eval/judges/build_set.py", "seed": SEED, "counts": counts,
           "questions_excluded": questions_excluded, "items": items}
    with open(os.path.join(HERE, "judge_set.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    write_sheet(items)
    print(json.dumps(counts), "questions excluded:", questions_excluded)


def write_sheet(items):
    """labeling_sheet.csv: every item with blank reviewer columns, for a person to fill in."""
    with open(os.path.join(HERE, "labeling_sheet.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "source", "kind", "label_by_construction", "perturbation_edit", "previous_line",
                    "line", "cited_sentences", "reviewer_verdict (supported/partly/unsupported/no_claim)",
                    "reviewer_notes"])
        for it in items:
            w.writerow([it["id"], it["source"], it["kind"], it["label"] or "", it.get("edit", ""), it["prev"],
                        it["text"], " | ".join(f"[{s}] {t}" for s, t in it["cited"]), "", ""])


if __name__ == "__main__":
    main()
