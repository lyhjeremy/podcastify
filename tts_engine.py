"""
tts_engine.py - synthesize a two-host script into one audio file with chapters.
==============================================================================
Each line is spoken by its speaker's voice (A / B) via Microsoft edge-tts (free,
no API key), four lines at a time. The per-line MP3s are joined with ffmpeg's
concat demuxer, then the outline's sections are written as chapters: ID3 CHAP
frames in an MP3, or chapter atoms in an M4B (AAC) when the output ends in .m4b.

    pip install edge-tts
    # ffmpeg and ffprobe on PATH for joining, chapters and durations
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile

import edge_tts

DEFAULT_VOICES = {"A": "en-US-AndrewNeural", "B": "en-US-AvaNeural"}


def find_ffmpeg(name: str = "ffmpeg") -> str | None:
    exe = shutil.which(name)
    if exe:
        return exe
    for c in [os.environ.get("FFMPEG_DIR", ""), r"C:\ffmpeg\bin", r"C:\Program Files\ffmpeg\bin",
              "/usr/bin", "/usr/local/bin", "/opt/homebrew/bin"]:
        p = os.path.join(c, name + (".exe" if os.name == "nt" else ""))
        if c and os.path.exists(p):
            return p
    return None


def duration(path: str) -> float | None:
    probe = find_ffmpeg("ffprobe")
    if not probe:
        return None
    r = subprocess.run([probe, "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
                       capture_output=True, text=True)
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except (KeyError, ValueError):
        return None


async def _render_all(jobs, rate, progress, limit=4):
    sem = asyncio.Semaphore(limit)
    done = 0

    async def one(text, voice, path):
        nonlocal done
        async with sem:
            for attempt in range(3):
                try:
                    await edge_tts.Communicate(text, voice, rate=rate).save(path)
                    break
                except Exception:  # edge-tts drops connections now and then
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2 + attempt * 3)
            done += 1
            if done % 10 == 0 or done == len(jobs):
                progress(f"  rendered {done}/{len(jobs)} lines")

    await asyncio.gather(*(one(*j) for j in jobs))


def _ffmeta(chapters: list[tuple[str, float, float]], title: str) -> str:
    out = [";FFMETADATA1", f"title={_esc(title)}", "artist=Podcastify"]
    for name, start, end in chapters:
        out += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(start * 1000)}", f"END={int(end * 1000)}",
                f"title={_esc(name)}"]
    return "\n".join(out) + "\n"


def _esc(s: str) -> str:
    return "".join("\\" + c if c in "=;#\\\n" else c for c in s)


def synthesize(lines, out_path: str, voices=None, rate: str = "+0%", sections=None,
               title: str = "Podcastify episode", progress=print) -> dict:
    """Render lines (dicts with speaker/text/section, or (speaker, text) tuples) to out_path.

    Returns {"path", "duration", "timings": [(start, dur) per line], "chapters": [...]}.
    """
    voices = voices or DEFAULT_VOICES
    lines = [l if isinstance(l, dict) else {"speaker": l[0], "text": l[1], "section": 0} for l in lines]
    lines = [l for l in lines if l["text"].strip()]
    if not lines:
        raise RuntimeError("Nothing to synthesize.")
    ff = find_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg is required (brew install ffmpeg | winget install Gyan.FFmpeg).")
    work = tempfile.mkdtemp(prefix="podcastify_")
    try:
        jobs = [(l["text"].strip(), voices.get(l["speaker"], DEFAULT_VOICES["A"]),
                 os.path.join(work, f"seg_{i:04d}.mp3")) for i, l in enumerate(lines)]
        asyncio.run(_render_all(jobs, rate, progress))
        durs = [duration(j[2]) or 0.0 for j in jobs]
        timings, t = [], 0.0
        for d in durs:
            timings.append((round(t, 3), round(d, 3)))
            t += d

        listfile = os.path.join(work, "list.txt")
        with open(listfile, "w", encoding="utf-8") as fh:
            fh.writelines(f"file '{j[2]}'\n" for j in jobs)
        joined = os.path.join(work, "joined.mp3")
        subprocess.run([ff, "-y", "-f", "concat", "-safe", "0", "-i", listfile, "-c", "copy", joined],
                       check=True, capture_output=True)
        total = duration(joined) or t

        chapters = []
        if sections:
            starts = {}
            for l, (st, _) in zip(lines, timings):
                starts.setdefault(l.get("section", 0), st)
            keys = sorted(starts)
            for k_i, k in enumerate(keys):
                end = starts[keys[k_i + 1]] if k_i + 1 < len(keys) else total
                name = sections[k]["title"] if k < len(sections) else f"Part {k + 1}"
                chapters.append((name, 0.0 if k_i == 0 else starts[k], end))
        meta = os.path.join(work, "meta.txt")
        with open(meta, "w", encoding="utf-8") as fh:
            fh.write(_ffmeta(chapters, title))

        if out_path.lower().endswith((".m4b", ".m4a")):
            cmd = [ff, "-y", "-i", joined, "-i", meta, "-map", "0:a", "-map_metadata", "1",
                   "-map_chapters", "1", "-c:a", "aac", "-b:a", "64k", "-f", "mp4", out_path]
        else:
            cmd = [ff, "-y", "-i", joined, "-i", meta, "-map", "0:a", "-map_metadata", "1",
                   "-map_chapters", "1", "-c", "copy", "-id3v2_version", "3", "-write_id3v1", "0", out_path]
        subprocess.run(cmd, check=True, capture_output=True)
        progress(f"  wrote {out_path} ({total / 60:.1f} min, {len(chapters)} chapters)")
        return {"path": out_path, "duration": round(total, 3), "timings": timings,
                "chapters": [{"title": c[0], "start": round(c[1], 3), "end": round(c[2], 3)} for c in chapters]}
    finally:
        shutil.rmtree(work, ignore_errors=True)
