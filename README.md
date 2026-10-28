# Podcastify

A Python command-line tool that turns any EPUB or article into a two-host podcast: one host narrates, a co-host interjects, and free neural voices render it to a single MP3 with a transcript.

[Live site](https://lyhjeremy.github.io/podcastify/) · [Write-up](https://lyhjeremy.github.io/podcastify/overview/)

Design and copy follow [these standards](https://github.com/lyhjeremy/lyhjeremy/blob/main/DESIGN_STANDARDS.md).

## Sample run

The first 1,800 characters of *Alice's Adventures in Wonderland* (Project Gutenberg), run on 2026-10-26 with edge-tts 7.3.1:

```
$ python podcastify.py alice-excerpt.txt
Source: alice-excerpt  (1,800 chars)
Transcript: alice-excerpt.transcript.md  (9 turns)
  [1/9] B: Welcome to Podcastify. Today we're getting into alice-excerp...
  [2/9] A: That's right. Let's dive in....
  [3/9] A: Alice was beginning to get very tired of sitting by her sist...
  ...
  [9/9] A: See you next time....
  wrote alice-excerpt.podcast.mp3

$ ffprobe -v error -show_entries format=duration alice-excerpt.podcast.mp3
duration=119.304000
```

One command produced 1 minute 59 seconds of two-voice audio and a Markdown transcript.

## Two modes

- `rule` (default, no API key): the narrator reads the source in three-sentence segments, and after every second segment the co-host adds a reaction and the narrator a bridge line, both drawn from fixed pools. Faithful to the source.
- `llm` (needs `ANTHROPIC_API_KEY`): sends up to 12,000 characters to the Anthropic Messages API and asks Claude to rewrite them as a host and co-host conversation, returned as a JSON array of turns. Falls back to `rule` without a key.

## How it works

1. `podcastify.py` reads `.epub` (ebooklib, BeautifulSoup), `.txt`, `.md` or `.html`.
2. `script_builder.py` builds a list of `(speaker, text)` turns and writes the transcript.
3. `tts_engine.py` renders each turn with Microsoft edge-tts (default voices en-US-AndrewNeural and en-US-AvaNeural) and joins the segments with ffmpeg's concat demuxer, or byte-joins them if ffmpeg is missing.

Stack: Python, edge-tts, ffmpeg, ebooklib, BeautifulSoup, Anthropic Messages API (over `urllib`, no SDK).

## Limitations

- In `rule` mode the co-host doesn't respond to the content, and over a long book its 12 reactions repeat.
- Sentences are split with a regular expression, which breaks on quoted speech: in the sample run the Rabbit's "Oh dear! Oh dear! I shall be late!" is cut across two turns.
- edge-tts uses an unofficial Microsoft endpoint. On 2026-10-26 version 7.2.8 got HTTP 403 and 7.3.1 worked, so keep it upgraded (`pip install -U edge-tts`).
- `llm` mode costs API usage and sends the source text to Anthropic.

## How to run

```bash
pip install -r requirements.txt
# ffmpeg recommended:  brew install ffmpeg  |  winget install Gyan.FFmpeg

python podcastify.py book.epub                 # book.podcast.mp3 + book.transcript.md
python podcastify.py article.txt --script-only # transcript only, no audio
python podcastify.py book.epub --max-chars 2000  # quick preview of the start
python podcastify.py book.epub --mode llm      # Claude rewrite (needs ANTHROPIC_API_KEY)
python podcastify.py book.epub --voice-a en-US-GuyNeural --voice-b en-US-JennyNeural
```

## Files

| File | What it is |
|---|---|
| `podcastify.py` | CLI: reads the source, builds the script, synthesizes audio |
| `script_builder.py` | Rule-based or Claude script builder |
| `tts_engine.py` | Per-turn edge-tts rendering and ffmpeg stitching |

`.gitignore` blocks `*.mp3` and transcripts, so books and audio stay out of the repo. The one exception is `overview/alice-excerpt.podcast.mp3`, the two-minute public-domain sample that plays on the write-up page. Use files you own or public-domain titles, for personal listening.

## License

[MIT](LICENSE) © 2026 Jeremy Lee
