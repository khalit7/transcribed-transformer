"""Taskmaster-1 (spoken) + Taskmaster-2 -> data/interim/taskmaster/{train,val}.jsonl (Track P, tier 3).

Input: data/raw/taskmaster/tm1-woz-dialogs.json and tm2-*.json. Only the spoken
Wizard-of-Oz dialogues are on disk; `self-dialogs.json` (typed prose in the same
schema) is refused by name here as a second line of defence.

Tier 3: human transcription of speech with partially repaired disfluencies and
no audio. These documents are NOT training text until a channel passes the gate
(SYNTHSHEET); they are prepared now for case construction and for the channel's
future input. Rendering: `SPEAKER_00` = ASSISTANT (advisor), `SPEAKER_01` = USER
(customer), the diarised-label convention with the role kept in meta rather than
shown; empty-speaker utterances with real text become `SPEAKER_02` rather than
being dropped, so line numbering stays faithful to the source.

Usage: uv run python -m src.preprocessing.taskmaster
"""

import json

from src.preprocessing.common import RAW, Document, Writer, render_turns

SPEAKER = {"ASSISTANT": "SPEAKER_00", "USER": "SPEAKER_01"}


def main() -> None:
    writer = Writer("taskmaster")
    empty = 0
    for path in sorted((RAW / "taskmaster").glob("*.json")):
        if "self-dialog" in path.name:
            raise SystemExit(f"refusing written self-dialogues: {path}")
        subset = path.stem
        for conv in json.loads(path.read_text()):
            turns = []
            for u in conv.get("utterances", []):
                text = (u.get("text") or "").strip()
                if not text:
                    continue
                turns.append((SPEAKER.get(u.get("speaker", ""), "SPEAKER_02"), text))
            if not turns:
                empty += 1
                continue
            cid = conv["conversation_id"]
            writer.add(
                Document(
                    source="taskmaster", doc_id=cid, track="track-p", tier=3, asr_system=None,
                    has_speakers=True, n_turns=len(turns),
                    n_words=sum(len(t.split()) for _, t in turns), text=render_turns(turns),
                    meta={"subset": subset, "instruction_id": conv.get("instruction_id", ""),
                          "roles": "SPEAKER_00=advisor,SPEAKER_01=customer"},
                ),
                cid,
            )
    print(f"empty dialogues dropped: {empty}")
    writer.close()


if __name__ == "__main__":
    main()
