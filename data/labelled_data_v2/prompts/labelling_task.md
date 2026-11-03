# Labelling task

You are labelling podcast episode transcripts for a research dataset. Each episode directory under `episodes/` holds
`transcript.txt` (one turn per line, numbered `n. speaker:text`) and `questions.json` (the questions to answer, each
with its text, a `description` of definitions that apply, and `options` with the criterion for each answer value). The
speaker is `host_1`, `host_2`, ... for the hosts and `guest_1`, `guest_2`, ... for the guests (the numbers only tell
people apart); any other voice (an advert, a played clip, a fragment) keeps its `SPEAKER_NN` diarisation tag.

For every episode assigned to you, answer every question in `questions.json` and write `labels.json` in the same
directory, a JSON object keyed by question id:

{"<question id>": {"evidence": [<line numbers>], "answer": "<one option value>", "summary": "<one or two sentences>",
"confidence": <0 to 1>}, ...}

Rules:
1. Read the whole transcript before answering; use Grep on `transcript.txt` to check every mention of a name, topic or
   phrase, and Read to see the lines around it. If Grep is not among your tools, load it first with ToolSearch
   (`select:Grep`); do not label without it. Decide strictly by the option criteria and the definitions; do not use
   knowledge from outside the transcript.
2. Evidence is the list of line numbers that bear on the answer. An answer of `none` (nothing of the kind happened)
   carries an empty evidence list. Cite the lines you actually relied on, including the later line when a question
   pairs an earlier event with a later outcome.
3. The summary states the reasoning in one or two sentences that refer to what was said.
4. Work only inside your assigned episode directories (absolute paths are given in your instructions). Do not read other
   episodes, the run's other files, other files in this repository, or anything else; do not modify `transcript.txt` or
   `questions.json`. Read and Grep to inspect (and ToolSearch only to load Grep), Write to produce `labels.json` and
   `notes.md`: no other tools and no shell commands. If Read refuses a long transcript, read it in consecutive parts with offset and limit until the last
   line.
5. Write `labels.json` for each episode as soon as that episode is done, so partial progress is saved; on a retry,
   overwrite the whole file. Optionally write `notes.md` beside it with anything you found hard to decide.
6. Answer every question; if a question's `answer` cannot be decided, pick the option whose criterion fits best and set
   `confidence` low, and say why in the summary. `confidence` is your own probability that the answer is right: 1.0
   means certain, 0.5 means a coin toss. Use the option values exactly as written. The summary may cite line numbers.
