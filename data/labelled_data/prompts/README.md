# Prompt templates for labelled data v1 (and the transfer task)

Prompt files loaded by the v1 code through `src/prompts.py` (`string.Template`, `$name` placeholders, one trailing
newline removed on load). Tracked in git and hashed by the review gate.

| File | Used by | Purpose |
|---|---|---|
| `train_task_compliance.txt` | `src/train/data.py` `task_prompt` (compliance families) | the trained models' prompt on the compliance data |
| `train_task_transfer.txt` | `src/train/data.py` `task_prompt` (family `transfer`, ECtHR) | the trained models' prompt on the transfer task |
| `label.txt` | `src/synthesis/label.py` | the v1 labellers (one question per call) |
| `identify_speakers.txt` | `src/synthesis/identify_speakers.py` | the v1 host identification (one host) |
| `grade_labels.txt` | `src/synthesis/grade_labels.py` | the judge |
| `probe_bank.txt` | `src/synthesis/probe_bank.py` | the answer-distribution probe of the bank |

The v2 pipeline's templates live in `data/labelled_data_v2/prompts/`.
