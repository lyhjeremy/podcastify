"""
grounding.py - check every script line against the source spans it cites.
=========================================================================
Two checks, then one repair:

local   sentence-transformers all-MiniLM-L6-v2 cosine similarity between the line
        and its cited sentences (the best of the joined span and each sentence),
        plus a number check: every figure written in digits in the line has to
        appear in the cited text. Cheap, runs offline, catches lines that cite
        the wrong place. A line is flagged below LOCAL_THRESHOLD (0.40, set
        before any eval run) or on a number mismatch.
llm     A separate `claude -p` call reads each line with its cited sentences and
        returns supported, partly, unsupported or no_claim. Lines the script
        writer marked as non-claims are checked too, against their two closest
        sentences in the section's evidence, so a factual line cannot skip the
        check by being labelled a reaction.
repair  Lines that come back partly or unsupported are sent once for a rewrite
        that the cited sentences fully support (or a drop). Rewrites are checked
        again; any that still fail are dropped. Every line keeps a record of what
        happened to it.

The LLM verdict decides; the local check is reported beside it.
"""
from __future__ import annotations

import re

import llm
import pipeline
import textspans

LOCAL_THRESHOLD = 0.40
LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def _encoder():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            _model = False
        else:
            _model = SentenceTransformer(LOCAL_MODEL)
    return _model


def _numbers(s: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", s)}


def local_check(line: str, cited: list[str]) -> dict:
    enc = _encoder()
    joined = " ".join(cited)
    missing = sorted(_numbers(line) - _numbers(joined))
    if not enc or not cited:
        return {"sim": None, "numbers_missing": missing, "flag": bool(missing) or not cited}
    vecs = enc.encode([line, joined] + cited, normalize_embeddings=True)
    sims = [float(vecs[0] @ v) for v in vecs[1:]]
    sim = round(max(sims), 3)
    return {"sim": sim, "numbers_missing": missing, "flag": sim < LOCAL_THRESHOLD or bool(missing)}


def nearest(line: str, candidates: list[tuple[str, str]], k: int = 2) -> list[str]:
    """Ids of the k candidate sentences closest to the line (MiniLM), or the first k."""
    enc = _encoder()
    if not candidates:
        return []
    if not enc:
        return [c[0] for c in candidates[:k]]
    vecs = enc.encode([line] + [c[1] for c in candidates], normalize_embeddings=True)
    sims = vecs[1:] @ vecs[0]
    order = sorted(range(len(candidates)), key=lambda i: -sims[i])
    return [candidates[i][0] for i in order[:k]]


CHECK_SYSTEM = ("You are a strict fact-checker for an audio script about a source text. You judge each "
                "line only against the source sentences shown with it, never against outside knowledge. "
                "Reply with JSON only.")
CHECK_PROMPT = """The source is "{title}" by {author}. For each numbered script line, compare what it says about the source with the source sentences
shown under it. The previous line is given for context only (a short answer may lean on a question).

Verdicts:
- "supported": everything the line asserts or presupposes about the source is stated in, or follows
  directly from, the shown sentences. Loose paraphrase, summary and a host's framing are fine.
- "partly": the main point is supported but at least one detail is not (an added detail, a wrong or
  unsupported number, a name, an overstatement).
- "unsupported": the main assertion is not in the shown sentences, or contradicts them.
- "no_claim": the line asserts nothing about the source's content: a greeting, the title and author,
  (naming the work, its author or what kind of text it is is never a claim),
  a reaction, an opinion clearly framed as the host's own, a transition, or a question that presupposes
  nothing.

Reply as JSON: {{"verdicts": [{{"n": 1, "verdict": "supported", "why": "under 15 words"}}]}}

LINES:
{items}
"""


def _render(items: list[dict]) -> str:
    out = []
    for it in items:
        out.append(f"### Line {it['n']} ({'Alex' if it['speaker'] == 'A' else 'Sam'})")
        if it.get("prev"):
            out.append(f"Previous line: {it['prev']}")
        out.append(f"Line: {it['text']}")
        out.append("Source sentences:")
        out += [f"  [{sid}] {t}" for sid, t in it["cited"]] or ["  (none cited)"]
        out.append("")
    return "\n".join(out)


def llm_check(items: list[dict], model=None, batch: int = 18, meta: dict | None = None) -> dict[int, dict]:
    """items: [{"n", "speaker", "prev", "text", "cited": [(id, text)]}] -> {n: {"verdict", "why"}}"""
    out: dict[int, dict] = {}
    for b in range(0, len(items), batch):
        chunk = items[b:b + batch]
        data = pipeline.cached("check", CHECK_PROMPT.format(
            items=_render(chunk), title=(meta or {}).get("title", "the source"),
            author=(meta or {}).get("author", "an unknown author")), CHECK_SYSTEM, model)
        for v in data.get("verdicts", []):
            verdict = str(v.get("verdict", "")).lower().strip()
            if verdict in ("supported", "partly", "unsupported", "no_claim"):
                out[int(v.get("n"))] = {"verdict": verdict, "why": v.get("why", "")}
    return out


REPAIR_SYSTEM = ("You fix lines in an audio script so that the source sentences fully support them. "
                 "Reply with JSON only.")
REPAIR_PROMPT = """Each line below failed a fact check against the source. Rewrite each one so that every
thing it says is stated in the source sentences given for it, keeping the same speaker, roughly the same
length and its place in the conversation (the previous line is shown). You may cite different sentences
from the candidates listed. If no faithful version would still make sense in the conversation, drop it.

Reply as JSON: {{"lines": [{{"n": 1, "text": "...", "src": ["S12"]}}, {{"n": 2, "drop": true}}]}}

{items}
"""


def check_script(lines: list[dict], text: str, sents: list[dict], shown: dict, repair: bool = True,
                 model=None, log=print, meta: dict | None = None) -> list[dict]:
    """Annotate each line with "check" and, when repair is on, rewrite or drop failing lines.
    Returns the full list (dropped lines kept, with "dropped": True)."""
    idx = textspans.index(sents)

    def cited(ln):
        return [(sid, text[idx[sid]["start"]:idx[sid]["end"]]) for sid in ln["src"] if sid in idx]

    # Unflagged lines get their two nearest evidence sentences so the checker can see them.
    for ln in lines:
        if not ln["src"]:
            cands = [(sid, text[idx[sid]["start"]:idx[sid]["end"]]) for sid in shown.get(ln["section"], [])]
            ln["auto_src"] = nearest(ln["text"], cands)

    def items_for(sel):
        out = []
        for n in sel:
            ln = lines[n]
            spans = cited(ln) or [(sid, text[idx[sid]["start"]:idx[sid]["end"]]) for sid in ln.get("auto_src", [])]
            out.append({"n": n + 1, "speaker": ln["speaker"], "text": ln["text"], "cited": spans,
                        "prev": lines[n - 1]["text"] if n else ""})
        return out

    for ln in lines:
        ln["check"] = {"local": local_check(ln["text"], [t for _, t in cited(ln)]) if ln["src"] else None}
    log(f"  grounding check: {len(lines)} lines")
    verdicts = llm_check(items_for(range(len(lines))), model, meta=meta)
    for n, ln in enumerate(lines):
        v = verdicts.get(n + 1, {"verdict": "unchecked", "why": "checker returned no verdict"})
        ln["check"].update(first=v["verdict"], why=v["why"], action="kept")
        ln["check"]["final"] = v["verdict"]

    failing = [n for n, ln in enumerate(lines) if ln["check"]["first"] in ("partly", "unsupported")]
    if not repair or not failing:
        return lines

    log(f"  repair: {len(failing)} lines")
    blocks = []
    for n in failing:
        ln = lines[n]
        cands = sorted(set(ln["src"]) | set(shown.get(ln["section"], [])),
                       key=lambda s: int(s[1:]))
        blocks.append("\n".join(
            [f"### Line {n + 1} ({'Alex' if ln['speaker'] == 'A' else 'Sam'})",
             f"Previous line: {lines[n - 1]['text'] if n else ''}",
             f"Failed line: {ln['text']}",
             f"Checker said: {ln['check']['first']}: {ln['check']['why']}",
             f"Cited: {', '.join(ln['src']) or 'none'}",
             "Candidate sentences:"] +
            [f"  [{sid}] {text[idx[sid]['start']:idx[sid]['end']]}" for sid in cands]))
    data = pipeline.cached("repair", REPAIR_PROMPT.format(items="\n\n".join(blocks)), REPAIR_SYSTEM, model)
    fixes = {int(f.get("n", 0)): f for f in data.get("lines", [])}

    recheck = []
    for n in failing:
        ln, f = lines[n], fixes.get(n + 1)
        ln["check"]["original"] = ln["text"]
        if not f or f.get("drop") or not (f.get("text") or "").strip():
            ln["check"]["action"] = "dropped"
            ln["check"]["final"] = "dropped"
            ln["dropped"] = True
            continue
        ln["text"] = f["text"].strip()
        ln["src"] = [s for s in f.get("src", []) if s in idx] or ln["src"]
        ln["claim"] = True
        ln["check"]["action"] = "rewritten"
        ln["check"]["local_after"] = local_check(ln["text"], [t for _, t in cited(ln)])
        recheck.append(n)

    if recheck:
        v2 = llm_check(items_for(recheck), model, meta=meta)
        for n in recheck:
            v = v2.get(n + 1, {"verdict": "unchecked", "why": "checker returned no verdict"})
            lines[n]["check"]["after"] = v["verdict"]
            lines[n]["check"]["why_after"] = v["why"]
            if v["verdict"] in ("supported", "no_claim"):
                lines[n]["check"]["final"] = v["verdict"]
            else:
                lines[n]["check"]["action"] = "dropped after rewrite"
                lines[n]["check"]["final"] = "dropped"
                lines[n]["dropped"] = True
    return lines


def rates(lines: list[dict]) -> dict:
    """First-pass and final verdict counts over the lines the checker saw as claims."""
    def count(key, pool):
        c = {}
        for ln in pool:
            v = ln["check"].get(key)
            c[v] = c.get(v, 0) + 1
        return c

    claims = [ln for ln in lines if ln["check"]["first"] in ("supported", "partly", "unsupported")]
    first = count("first", claims)
    n = len(claims) or 1
    actions = count("action", lines)
    local_flags = sum(1 for ln in claims if (ln["check"].get("local") or {}).get("flag"))
    agree = sum(1 for ln in claims if ln["check"].get("local") is not None and
                (ln["check"]["local"]["flag"]) == (ln["check"]["first"] != "supported"))
    with_local = sum(1 for ln in claims if ln["check"].get("local") is not None)
    return {
        "lines": len(lines),
        "claim_lines": len(claims),
        "unflagged_claims": sum(1 for ln in claims if not ln["src"] and ln.get("auto_src")),
        "first_pass": first,
        "unsupported_rate_first": round(first.get("unsupported", 0) / n, 3),
        "not_supported_rate_first": round((first.get("unsupported", 0) + first.get("partly", 0)) / n, 3),
        "actions": actions,
        "local_flags": local_flags,
        "local_llm_agreement": round(agree / with_local, 3) if with_local else None,
    }
