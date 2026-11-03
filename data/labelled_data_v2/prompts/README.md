# Prompt templates for labelled data v2

Every prompt a model sees in the v2 pipeline is a file in this folder, not a string in the code. The code loads them
through `src/prompts.py`. Files with `$name` placeholders are rendered with `string.Template` (a literal dollar sign
would be `$$`); `definitions.txt` and `labelling_task.md` are read verbatim, no substitution. A template file ends with
one newline, which the loader removes. The templates hold each prompt's frame; the per-call fragments (a question block,
the tag rule, the JSON shape of a multi-question reply) are assembled in the calling code. These files are tracked in git (an exception to the rule that nothing
under `data/labelled_data_v2/` is tracked) and are part of the review gate's hashed state, because a changed prompt
changes a run as much as changed code does.

| File | Used by | Model | Purpose |
|---|---|---|---|
| `train_task.txt` | `src/train/data.py` `task_prompt` (family `podcast`) | the fine-tuned arms | the prompt the trained models are given for a podcast question |
| `identify_speakers.txt` | `src/synthesis_v2/identify_hosts.py` | Sonnet | kind of recording, host tags, guest tags, same-person groups |
| `labelling_task.md` | `src/synthesis_v2/agent_label.py` (`TASK.md` in every run) | Opus subagents | the gold labellers' brief |
| `definitions.txt` | `src/synthesis_v2/question_bank.py` (`VOCAB`; `with_definitions` opens a question's context with it in every one-question prompt, the lean labeller prints it once) | every labeller and the arms | the shared definitions. Stored only here since 2026-11-02: `questions.jsonl` and new records hold each question's own description, so a reader of the questions needs this file too to rebuild any prompt |
| `label_episodes.txt` | `src/synthesis_v2/label_episodes.py` | Opus and Sonnet (`claude -p`, the pilot route) | several questions per call |
| `label_episodes_lean.txt` | `src/synthesis_v2/label_episodes.py` `prompt_lean` (`--lean`) | Opus (`claude -p`, the production route, D82) | every question of an episode in one call, the shared definitions printed once (`$definitions`) |
| `labeller_system.txt` | `src/synthesis_v2/label_episodes.py` (`--lean`, via `llm.claude_command`) | Opus | replaces Claude Code's system prompt on the lean route (no tools, caching off) |

The v1 pipeline's templates live in `data/labelled_data/prompts/` in the same form.
