# Podcastify

A Python command-line tool that turns an EPUB or article into a two-host podcast. In LLM mode it reads the whole source, writes a host and co-host conversation in which every factual line cites the source sentences it rests on, checks each line against those sentences, and renders an MP3 (or M4B) with a chapter per section. Rule mode, which needs no model, reads the source aloud in two voices.

[Live site](https://lyhjeremy.github.io/podcastify/) · [Write-up](https://lyhjeremy.github.io/podcastify/overview/) · [Narrative](writeup.md)

Design and copy follow [these standards](https://github.com/lyhjeremy/lyhjeremy/blob/main/DESIGN_STANDARDS.md).

## Results (October 2026 eval)

Four sources, each run three ways at a six-minute target: rule mode reading from the start until it has used as many words as the LLM episode, the LLM draft before the grounding pass, and the LLM episode after it. Key points are 12 per source, written by Opus from the full text and committed before any episode was generated (`eval/keypoints/`); Opus graded each script (covered 1, partly 0.5). "Not fully supported" is the share of claim lines that Opus, as a second checker outside the pipeline, rated partly or unsupported against the cited sentences.

| Source | Mode | Key points (of 12) | Not fully supported | Length | Words/min |
|---|---|---|---|---|---|
| Great Stink (Wikipedia) | rule | 8.0 | verbatim | 8:22 | 160 |
| | LLM draft | 10.5 | 7.8% (4 of 51) | 8:17 | 154 |
| | LLM checked | 10.0 | 6.5% (3 of 46) | 7:53 | 154 |
| Civil Disobedience (Thoreau) | rule | 4.5 | verbatim | 5:49 | 176 |
| | LLM draft | 8.0 | 3.0% (1 of 33) | 5:40 | 169 |
| | LLM checked | 8.5 | 3.2% (1 of 31) | 5:35 | 168 |
| Origin of Species, ch. III (Darwin) | rule | 4.0 | verbatim | 7:08 | 165 |
| | LLM draft | 5.0 | 6.5% (3 of 46) | 6:15 | 160 |
| | LLM checked | 5.0 | 6.5% (3 of 46) | 6:16 | 160 |
| Alice's Adventures in Wonderland (whole book) | rule | 1.0 | verbatim | 6:07 | 193 |
| | LLM draft | 6.0 | 0.0% (0 of 42) | 5:55 | 173 |
| | LLM checked | 6.0 | 2.4% (1 of 42) | 5:55 | 173 |

- Coverage: the LLM episodes averaged 7.4 of 12 key points; rule mode, 4.4. The gap is widest on the longest source, where six minutes of rule mode do not reach the end of chapter one.
- The pipeline's own checker (Sonnet) failed 9 of 151 claim lines on the first pass, all as "partly". Three were rewritten and kept, six dropped.
- The second checker found 8 of 172 claim lines not fully supported before the repair and 8 of 165 after. The repair removes what the first checker flags, and the two checkers mostly flag different lines, so the grounding pass did not lower the second checker's rate.
- Listenability, judged blind by Opus in both orders: LLM beat rule mode on all four sources. The unchecked draft beat the checked episode on three of four (the fourth was identical), because a dropped line can leave a question without its answer.
- Hand check: I read 20 random claim lines from the drafts against their cited sentences before looking at any verdict. My reading agreed with the pipeline's checker on pass or fail for 18 of 20, and with Opus for 19. Only one of the 20 failed by my reading, so this shows the checker passes good lines; it does not measure how many bad ones it misses.

Four sources and one run each: treat the grounding counts (single digits) as indicative. Full results: `eval/results/summary.json`; every line with its spans and verdicts: `eval/results/<source>.lines.json`.

## How LLM mode works

1. Spans. `textspans.py` splits the source into numbered sentences with character offsets (`S1`, `S2`, ...). These are the only things a line can cite.
2. Map. `pipeline.py` sends each chunk of about 10,000 characters, sentences numbered, to the model, four at a time. It returns key points that cite sentence ids, with an importance from 1 to 3. The whole book is read, not the first 12,000 characters as before.
3. Reduce. The model picks about two points per target minute, in source order, grouped into three to five titled sections.
4. Script. The model writes the conversation from the outline and the cited sentences. Each line is JSON: speaker, text, whether it makes a claim, and the sentence ids it rests on.
5. Check (`grounding.py`). Local: all-MiniLM-L6-v2 cosine between the line and its cited sentences (flag below 0.40, set before the eval) and a check that every figure in digits appears in the cited text. LLM: a separate `claude -p` call returns supported, partly, unsupported or no claim for every line, including lines the writer marked as non-claims (checked against their two nearest outline sentences). Lines rated partly or unsupported are rewritten once and checked again, and dropped if they still fail.
6. Audio (`tts_engine.py`). edge-tts renders each line, four at a time; ffmpeg joins them and writes the outline sections as ID3 chapters in the MP3, or chapter atoms in an M4B.

The transcript (`<name>.transcript.md`) shows each line's sentence ids and character offsets, its verdict, and any rewrite or drop. `<name>.episode.json` has everything: outline, lines, spans, checks and timings.

## How to run

```bash
pip install -r requirements.txt          # edge-tts, ebooklib, beautifulsoup4, sentence-transformers
# ffmpeg and ffprobe on PATH:  brew install ffmpeg  |  winget install Gyan.FFmpeg

python podcastify.py book.epub                       # rule mode: two-voice reading, no LLM
python podcastify.py book.epub --mode llm            # grounded conversation, 8-minute target
python podcastify.py book.epub --mode llm --minutes 12 -o book.m4b
python podcastify.py article.txt --mode llm --script-only   # transcript + episode JSON, no audio
python podcastify.py article.txt --mode llm --no-check      # skip the grounding pass
```

LLM backends, in order:

- The Claude CLI (`claude -p`), signed in with a Claude subscription. No API key. Every call runs with `--tools "" --setting-sources ""`, a fixed system prompt, no saved session and a temp working directory, so the model sees the prompt and nothing on disk. This is how the repo was built and evaluated.
- The Anthropic Messages API, used automatically when `ANTHROPIC_API_KEY` is set (over `urllib`, no SDK). Force either with `LLM_BACKEND=cli` or `LLM_BACKEND=api`.

The model defaults to Sonnet; change it with `--model opus` or `PODCASTIFY_MODEL`. Replies are cached by prompt hash in `~/.cache/podcastify` (or `PODCASTIFY_CACHE`), so a rerun after a failure resumes. Without either backend, `--mode llm` falls back to rule mode.

To rerun the eval (about 130 model calls; sources and audio go to `~/data-cache/podcastify/`):

```bash
python eval/fetch_sources.py                  # download and clean the four sources
python eval/make_keypoints.py                 # only if eval/keypoints/ is missing; it is frozen
PODCASTIFY_CACHE=~/data-cache/podcastify/llm-cache python eval/run_eval.py
python eval/hand_check.py sample              # 20 lines, verdicts hidden; fill eval/hand_check.json
python eval/hand_check.py score
python eval/export_web.py alice               # overview/episode.json, eval.json and the sample MP3
```

## Rule mode

The default. The narrator reads three sentences at a time; after every second segment the co-host adds a reaction and the narrator a bridge line, both from fixed pools. It is the source verbatim, it costs nothing, and the co-host responds to nothing in particular. The 2026-10-26 sample run (1,800 characters of *Alice* as 1 minute 59 seconds of audio) is described in [writeup.md](writeup.md).

## Limitations

- Four sources, one run each. The coverage gap is large; the grounding counts are small and would move by a few lines on another run.
- The key points, coverage grading, second check and listenability judge are all Claude models, as is the script writer, so the judge may favor writing like its own.
- A "supported" verdict means the cited sentences back the line. Nothing checks what the outline leaves out: six minutes of *Alice* skip the pool of tears, the Caucus-race and the croquet game.
- The checkers make reading mistakes. In *Alice*'s last paragraph both misread who "this same little sister of hers" is, and a correct line was dropped.
- Episode length runs loose: 5.6 to 7.9 minutes against a six-minute target.
- Sentence splitting is a regular expression; quoted speech like "Oh dear! Oh dear!" becomes two spans.
- edge-tts uses an unofficial Microsoft endpoint. On 2026-10-26 version 7.2.8 got HTTP 403 and 7.3.1 worked, so keep it upgraded.
- LLM mode sends the source text to Anthropic (through the CLI or the API), and every mode sends each line to Microsoft's edge-tts service.

## Files

| File | What it is |
|---|---|
| `podcastify.py` | CLI: reads the source, builds the script, checks it, synthesizes audio |
| `script_builder.py` | Rule mode, the LLM entry point, transcript Markdown |
| `textspans.py` | Sentence spans with character offsets, chunking |
| `pipeline.py` | Map, reduce and script passes, with a disk cache |
| `grounding.py` | Local and LLM checks, one repair pass, rates |
| `llm.py` | `claude -p` wrapper with the API fallback |
| `tts_engine.py` | edge-tts rendering, ffmpeg join, ID3 or M4B chapters |
| `eval/` | Source fetcher, frozen key points, eval runner, hand check, web export, results |
| `overview/` | The write-up page, its data, and the one committed sample (`alice.episode.mp3`, 1.8 MB) |

`.gitignore` blocks audio, transcripts and episode JSON, so books and audio stay out of the repo except the sample above. Use files you own or public-domain titles, for personal listening.

## Credits

Eval sources: *Alice's Adventures in Wonderland* (Lewis Carroll), *On the Duty of Civil Disobedience* (Henry David Thoreau) and *On the Origin of Species*, first edition, chapter III (Charles Darwin), from Project Gutenberg, public domain. [Great Stink](https://en.wikipedia.org/w/index.php?title=Great_Stink&oldid=1381900566), revision 1381900566, by Wikipedia contributors, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); sentences from it are quoted in `eval/results/hand_sample.md` and paraphrased in `eval/results/greatstink.lines.json`, under the same license.

## License

Code: [MIT](LICENSE) © 2026 Jeremy Lee.
