"""
llm.py - one text-completion call, through the local Claude CLI or the API.
===========================================================================
Backend, in priority order:

    cli  The local `claude` binary (Claude Code), signed in with a Claude
         subscription. No API key, no per-token bill. This is the path the repo
         was built and evaluated with.
    api  The Anthropic Messages API over urllib (no SDK). Used automatically
         when ANTHROPIC_API_KEY is set.

Force one with LLM_BACKEND=cli or LLM_BACKEND=api. Choose the model with
PODCASTIFY_MODEL (a CLI alias such as "sonnet" or "opus", or a full model id).

The CLI runs isolated: built-in tools off (--tools ""), no user or project
settings and no CLAUDE.md (--setting-sources ""), our system prompt in place of
Claude Code's, no saved session, and a neutral working directory. Without that,
`claude -p` could read files off disk or pick up the user's own instructions,
and the grounding check would no longer be checking against the source alone.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request

DEFAULT_MODEL = os.environ.get("PODCASTIFY_MODEL", "sonnet")
API_MODELS = {"sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5", "haiku": "claude-haiku-4-5"}


class LLMError(RuntimeError):
    pass


def _which_claude() -> str | None:
    return shutil.which("claude.cmd") or shutil.which("claude.exe") or shutil.which("claude")


def backend() -> str:
    forced = os.environ.get("LLM_BACKEND", "").strip().lower()
    if forced in ("cli", "api"):
        return forced
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "api"
    if _which_claude():
        return "cli"
    raise LLMError("No LLM backend: install the Claude CLI (claude.com/claude-code) "
                   "or set ANTHROPIC_API_KEY.")


def available() -> bool:
    try:
        backend()
        return True
    except LLMError:
        return False


def complete(prompt: str, system: str, model: str | None = None,
             max_tokens: int = 8000, timeout: int = 600, retries: int = 2) -> str:
    """Return the model's text reply. Retries transient failures."""
    model = model or DEFAULT_MODEL
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if backend() == "cli":
                return _cli(prompt, system, model, timeout)
            return _api(prompt, system, model, max_tokens, timeout)
        except (LLMError, OSError) as exc:  # noqa: PERF203
            last = exc
            time.sleep(5 * (attempt + 1))
    raise LLMError(str(last))


def _cli(prompt: str, system: str, model: str, timeout: int) -> str:
    exe = _which_claude()
    if not exe:
        raise LLMError("`claude` CLI not found on PATH.")
    cmd = [exe] if not (os.name == "nt" and exe.lower().endswith((".cmd", ".bat"))) else ["cmd", "/c", exe]
    cmd += ["-p", "--tools", "", "--setting-sources", "", "--no-session-persistence",
            "--output-format", "json", "--system-prompt", system, "--model", model]
    try:
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, cwd=tempfile.gettempdir())
    except subprocess.TimeoutExpired as exc:
        raise LLMError(f"claude CLI timed out after {timeout}s") from exc
    if r.returncode != 0:
        raise LLMError(f"claude CLI failed: {(r.stderr or r.stdout)[:300]}")
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError as exc:
        raise LLMError(f"claude CLI returned non-JSON: {r.stdout[:200]}") from exc
    if data.get("is_error"):
        raise LLMError(f"claude CLI error: {str(data.get('result'))[:300]}")
    return (data.get("result") or "").strip()


def _api(prompt: str, system: str, model: str, max_tokens: int, timeout: int) -> str:
    body = json.dumps({
        "model": API_MODELS.get(model, model),
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                 "anthropic-version": "2023-06-01", "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return "".join(b.get("text", "") for b in data.get("content", [])).strip()


def complete_json(prompt: str, system: str, model: str | None = None, **kw):
    """complete() and parse the first JSON object or array in the reply.
    One retry with a reminder if the reply does not parse."""
    for attempt in range(2):
        raw = complete(prompt if attempt == 0 else
                       prompt + "\n\nYour previous reply was not valid JSON. Reply with JSON only.",
                       system, model, **kw)
        try:
            return parse_json(raw)
        except ValueError:
            continue
    raise LLMError(f"Reply did not parse as JSON: {raw[:300]}")


def parse_json(raw: str):
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = raw.find(opener), raw.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(raw[i:j + 1])
            except json.JSONDecodeError:
                pass
    raise ValueError("no JSON found")


if __name__ == "__main__":
    print(f"[backend={backend()} model={DEFAULT_MODEL}]")
    print(complete("Reply with exactly one word: PONG", "You are terse."))
