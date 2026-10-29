# Podcastify

Podcastify turns an EPUB or article into a two-host podcast: Alex, who has read the source, and Sam, who asks the questions a listener would. It started in June 2026 as a two-voice reader and was rebuilt in October 2026 around one question: can an LLM-written audio overview stay faithful to its source, and show a listener where each line came from?

## The original tool (June 2026)

The first version had two modes. Rule mode split the text into sentences with a regular expression, gave the narrator three at a time, and after every second segment had the co-host interject from a pool of twelve reactions ("Let me make sure I'm following") with a narrator bridge from a pool of five. It was faithful, because it was the source read aloud, and the co-host responded to nothing.

LLM mode sent the first 12,000 characters of the source to the Anthropic Messages API and asked for a conversation as a JSON array of turns, with an instruction not to invent facts. That needed an API key, and for a book it meant the episode was about the first chapter or two. Nothing checked the result.

The sample run on 2026-10-26: the first 1,800 characters of *Alice's Adventures in Wonderland* in rule mode became nine turns and 119.3 seconds of audio (measured with ffprobe), with the Rabbit's "Oh dear! Oh dear! I shall be late!" split across two turns by the sentence regex. That sample played on the write-up page until 2026-10-29; it is still in the repository history as `overview/alice-excerpt.podcast.mp3`.

## October 2026 upgrade

Written 2026-10-29. Three things changed: the LLM path reads the whole source and cites it, every line is checked against what it cites, and there is an eval that compares rule mode with LLM mode on sources of different kinds.

### Why

An LLM conversation is easier to listen to than a two-voice reading, and the eval below bears that out. The cost is trust: a listener cannot tell which lines are the source and which are the model. The fix tried here is to make every claim traceable to a span of the source by character offset, check it automatically, and show the result.

The LLM calls go through the local `claude -p` CLI on a Claude subscription instead of the API, since Jeremy has no API key. Each call runs with built-in tools off and no user settings loaded (`--tools "" --setting-sources ""`), the lesson from Project 22, where an agent run without them read files off disk. The API remains as an automatic fallback when `ANTHROPIC_API_KEY` is set.

### What was built

The source is split into numbered sentences with character offsets; these are the spans every citation points to. A map pass sends each chunk of about 10,000 characters to Sonnet and gets back key points with the sentence ids they rest on (169 for the whole of *Alice*). A reduce pass picks about two points per target minute, in order, in three to five titled sections. A script pass writes the conversation from that outline and the cited sentences, as JSON lines that each say whether they make a claim and which sentences back it.

The grounding check then looks at every line. The local half embeds the line and its cited sentences with all-MiniLM-L6-v2 and flags a similarity below 0.40, or a number in the line that the cited text lacks. The LLM half is a separate `claude -p` call that rates each line supported, partly, unsupported or no claim, given the line before it for context. Lines the writer marked as non-claims are checked too, against their two nearest outline sentences, so a factual line cannot dodge the check by being labeled a reaction; seven lines across four episodes were such unmarked claims. Lines rated partly or unsupported get one rewrite, are checked again, and are dropped if they still fail. The transcript records each verdict, rewrite and drop.

The audio side is the same edge-tts voices, now rendered four lines at a time, with the outline's sections written as ID3 chapter frames in the MP3, or as chapter atoms when the output is an M4B.

### The eval

Four sources of different kinds: the whole of *Alice* (143,485 characters), Thoreau's *Civil Disobedience* (51,135), chapter III of Darwin's *Origin of Species* (35,685) and the Wikipedia article on the Great Stink (26,640, revision 1381900566, CC BY-SA 4.0). Before generating anything, Opus wrote 12 key points per source from the full text; I checked the figures in them against the sources and committed them. Each source was then run three ways at a six-minute target: the full LLM pipeline, the same LLM draft before the grounding pass, and rule mode reading from the start until it had used as many words as the LLM episode.

Coverage, graded by Opus against the frozen points: the LLM episodes averaged 7.4 of 12 and rule mode 4.4. The gap is largest on *Alice* (6 against 1), where six minutes of reading aloud do not reach the end of chapter one. Darwin is the weak case for the LLM at 5 of 12: the outline chose Darwin's general statements of the argument and left out the worked examples the key-point list counts: the elephant arithmetic, the animals run wild in South America and Australia, the slugs and the weed seedlings, the cats, mice, bees and clover.

Grounding: the pipeline's checker failed 9 of 151 claim lines on the first pass, all "partly", and never "unsupported". Three were rewritten and kept; six were dropped. The failures were rarely invented facts. They were questions that presupposed something the cited sentences do not say ("It sits right on the river"), a quote attributed to a source the cited sentence does not name, and inferences stated as fact. To measure the effect without grading the pipeline with its own checker, Opus checked every line of both the draft and the final episode. It found 8 of 172 claim lines not fully supported in the drafts and 8 of 165 after the repair. The repair removed what Sonnet flagged, and Opus mostly flagged other lines. So the pass did not make the episodes measurably more faithful by the second checker's standard; what it adds is a per-line record a reader can check.

The checkers also make mistakes. At the end of *Alice*, the sister pictures "this same little sister of hers" grown up, which is Alice. Sonnet read it as the sister picturing herself, marked a correct line "partly", and the line was dropped after its rewrite; Opus made the same misreading of the next line.

Listenability, judged blind by Opus with each pair shown in both orders: the LLM scripts beat rule mode on all four sources in both orders. The unchecked draft beat the checked episode on three of four, and the fourth pair was identical because nothing failed. Dropping a line can leave Sam's question without Alex's answer, and the judge noticed.

Length: the LLM episodes ran 5.6 to 7.9 minutes against the six-minute target, at 154 to 173 words a minute; rule mode ran 160 to 193 words a minute, fastest on *Alice*'s short dialogue sentences.

Hand check: 20 claim lines drawn at random from the drafts, read against their cited sentences before looking at any verdict. My reading agreed with the pipeline's checker on pass or fail for 18 and with Opus for 19. One disagreement was the *Alice* ending above; in the other the checker passed "That points at the water, not the air", an inference the cited sentences about John Snow's pump do not make. Only one of the 20 lines failed by my reading, so the sample shows the checker passes good lines and says little about how many bad lines it misses.

### What the numbers do not show

Four sources and one run each. The coverage result is large and consistent across them; the grounding counts are single digits and would move on a rerun. Every judge in the eval is a Claude model, and so is the writer. A supported line is backed by its cited sentences, which is not the same as a fair summary: the outline decides what is left out, and six minutes of *Alice* skip the pool of tears, the Caucus-race and the croquet game. Nothing checks that.

### Superseded

- "LLM mode sends up to 12,000 characters": replaced on 2026-10-29 by the map, reduce and script pipeline over the whole source.
- "LLM mode needs `ANTHROPIC_API_KEY`": since 2026-10-29 it runs through `claude -p` by default; the key is an optional path.
- The 1 minute 59 second rule-mode sample on the write-up page: replaced on 2026-10-29 by the six-minute LLM episode of the whole of *Alice*, with its transcript, spans and verdicts.
