"""
tts_engine.py - synthesize a two-host script into a single podcast audio file.
=============================================================================
Each turn is spoken by the voice assigned to its speaker (A / B) via Microsoft
edge-tts (free, no API key). Per-turn MP3 segments are rendered, then stitched
into one file with ffmpeg. ffmpeg is also used for the optional .m4b.

    pip install edge-tts
    # ffmpeg on PATH (or discoverable) for concatenation / .m4b
"""
from __future__ import annotations
import os
import re
import shutil
import asyncio
import subprocess
import tempfile

import edge_tts

# Two natural, conversational US voices by default.
DEFAULT_VOICES = {"A": "en-US-AndrewNeural", "B": "en-US-AvaNeural"}


def find_ffmpeg() -> str | None:
    """Locate ffmpeg on PATH or in common install locations."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    candidates = [
        os.environ.get("FFMPEG_DIR", ""),
        r"C:\ffmpeg\bin", r"C:\Program Files\ffmpeg\bin",
        "/usr/bin", "/usr/local/bin", "/opt/homebrew/bin",
    ]
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    for c in candidates:
        if c and os.path.exists(os.path.join(c, name)):
            return os.path.join(c, name)
    return None


async def _say(text: str, voice: str, path: str, rate: str = "+0%") -> None:
    await edge_tts.Communicate(text, voice, rate=rate).save(path)


def synthesize(turns, out_path: str, voices=None, rate: str = "+0%",
               keep_segments: bool = False, progress=print) -> str:
    """Render `turns` (list of (speaker, text)) to a single MP3 at out_path."""
    voices = voices or DEFAULT_VOICES
    ff = find_ffmpeg()
    workdir = tempfile.mkdtemp(prefix="podcastify_")
    seg_paths = []
    try:
        for i, (spk, text) in enumerate(turns):
            text = text.strip()
            if not text:
                continue
            seg = os.path.join(workdir, f"seg_{i:04d}.mp3")
            voice = voices.get(spk, DEFAULT_VOICES["A"])
            progress(f"  [{i+1}/{len(turns)}] {spk}: {text[:60]}...")
            asyncio.run(_say(text, voice, seg, rate=rate))
            seg_paths.append(seg)

        if not seg_paths:
            raise RuntimeError("Nothing to synthesize.")

        if ff:
            listfile = os.path.join(workdir, "list.txt")
            with open(listfile, "w", encoding="utf-8") as fh:
                for p in seg_paths:
                    fh.write(f"file '{p}'\n")
            subprocess.run(
                [ff, "-y", "-f", "concat", "-safe", "0", "-i", listfile,
                 "-c", "copy", out_path],
                check=True, capture_output=True,
            )
        else:
            # Fallback: byte-concatenate MP3s (works for most players).
            progress("  ffmpeg not found - byte-concatenating segments.")
            with open(out_path, "wb") as out:
                for p in seg_paths:
                    with open(p, "rb") as fh:
                        out.write(fh.read())
        progress(f"  wrote {out_path}")
        return out_path
    finally:
        if not keep_segments:
            shutil.rmtree(workdir, ignore_errors=True)
