# Podcastify

Turn any **EPUB or article** into a **two-host podcast** you can listen to. One
host narrates, a co-host chimes in with reactions and questions, and the whole
thing is voiced with free, natural neural voices. Optionally, let Claude rewrite
the source into a genuine back-and-forth conversation.

> 🌐 **Overview:** https://lyhjeremy.github.io/podcastify/

## Why
Audiobooks are one voice reading at you. A two-host show is easier to listen to on
a walk or commute. Podcastify takes text you already have and stages it for two
voices, so it feels like a conversation about the material instead of a recital.

## Two modes
- **`rule`** (default, no API key) — a structured two-voice reading: the main host
  narrates the source in short segments, the co-host interjects between them.
  Faithful to the source text, just dramatised for two voices.
- **`llm`** (optional, set `ANTHROPIC_API_KEY`) — Claude rewrites the source into a
  natural host/co-host conversation.

## Quick start
```bash
pip install edge-tts ebooklib beautifulsoup4
# ffmpeg recommended for clean stitching:  winget install Gyan.FFmpeg  |  brew install ffmpeg

python podcastify.py book.epub                 # -> book.podcast.mp3 + transcript
python podcastify.py article.txt --script-only # transcript only, no audio
python podcastify.py book.epub --mode llm      # natural conversation (needs ANTHROPIC_API_KEY)
python podcastify.py book.epub --voice-a en-US-GuyNeural --voice-b en-US-JennyNeural
```

## Files
| File | What it is |
|---|---|
| `podcastify.py` | CLI: read an EPUB/text file, build the script, synthesize audio |
| `script_builder.py` | Turns source text into a two-host script (rule-based or Claude) |
| `tts_engine.py` | Renders each turn with edge-tts and stitches it with ffmpeg |

## Notes
This repo ships **code only** — no books and no audio. Point it at a file you own
or a public-domain title; output is for personal use. Tested end to end: a short
sample renders a valid two-voice MP3 via edge-tts + ffmpeg.

## License
[MIT](LICENSE) © 2026 Jeremy Lee
