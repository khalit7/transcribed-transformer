"""Export the releasable labelled data: the Track P records of the frozen splits, with a licence notice.

    uv run python -m src.synthesis.export --track p [--out data/labelled_data/release-p]

Reads the split files (`train.jsonl`, `val.jsonl`, `benchmark.jsonl`, so the benchmark records carry the teacher label,
the Opus label where there is one, cell and provenance) rather than the raw ledger, keeps the Track P records, marks each with its split, and
strips the billing fields (`generation_info.cost_usd`, `generation_info.claude_account`) that belong to this machine
and not to a release. Writes `labelled_data.jsonl`, `questions.jsonl` (the bank plus the questions written for these
calls) and `NOTICE.md` (the licences and attributions of the source corpora; the release is share-alike because AppTek
is). Only Track P can be exported: Track NC contains SPoRC text whose gate forbids redistribution.
"""

import argparse
import json
from pathlib import Path

from src.synthesis.question_bank import OUT_DIR, write_questions

SPLITS = ("train", "val", "benchmark")
BILLING = ("cost_usd", "claude_account")
NOTICE = """# Licence and attribution

This release is derived from three public corpora and is distributed as a whole under **CC BY-SA 4.0**
(https://creativecommons.org/licenses/by-sa/4.0/legalcode), because one of
them (AppTek Call-Center Dialogues) is share-alike and the ASR channel model used to make the "messy" variants of the
other two was fitted on it. Attribute as below when you use it.

| Records with `dataset` | Source | Licence | Attribution |
|---|---|---|---|
| `apptek` | AppTek Call-Center Dialogues, https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues (arXiv:2604.27543) | CC BY-SA 4.0 | AppTek. Verbatim segments and the recogniser output of a telephone-degraded copy of the audio. The source card states an evaluation-only intent; this release uses its calls as labelled training, validation and benchmark pairs for research-only fine-tuning (see the repository's DATASHEET), which a user of this release should weigh against that intent. |
| `taskmaster` | Taskmaster-1 and Taskmaster-2, https://github.com/google-research-datasets/Taskmaster | CC BY 4.0 | Google Research. The "messy" variant is synthetic: an ASR channel model applied to the human transcription. |
| `aci_bench` | ACI-Bench, https://github.com/microsoft/clinical_visit_note_summarization_corpus | CC BY 4.0 | Microsoft. The "messy" variant is synthetic as above; records built from 2026-09-29 on carry the corpus's own subset (aci, virtassist, virtscribe) in `meta.subset`, earlier ones only `meta.split_file`. |

Labels were produced by LLM labelling as described in the repository's `src/synthesis/README.md`; they are
model-generated, not human ground truth. `label` is the training labeller's own label on every record (a local 27B model
for general_qa, Claude Sonnet for the rare-event families). Benchmark records also carry `opus_label` where Claude Opus
labelled the pair as a second, independent labeller (a selected subset: the general_qa pairs where the training labeller
and a second local labeller disagreed, four with no second opinion, and the family pairs the training labeller called `fail`), `provenance.labels`
(every labeller's answer) and `provenance.opus_labeller`. Questions and their grading rules are in `questions.jsonl`. Each record's `split` says whether it is a
training, validation or benchmark pair of the repository's frozen split.
"""


def release_record(r: dict, split: str) -> dict:
    out = {**r, "split": split}
    out["generation_info"] = {k: v for k, v in r["generation_info"].items() if k not in BILLING}
    return out


def export_release(out: Path, src_dir: Path = OUT_DIR, track: str = "track-p") -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    counts = {s: 0 for s in SPLITS}
    with (out / "labelled_data.jsonl").open("w") as f:
        for split in SPLITS:
            path = src_dir / f"{split}.jsonl"
            if not path.exists():
                raise SystemExit(f"{path} is missing: build the splits (src.synthesis.benchmark) before exporting")
            for line in path.open():
                r = json.loads(line)
                if r["track"] == track:
                    f.write(json.dumps(release_record(r, split), ensure_ascii=False) + "\n")
                    counts[split] += 1
    (out / "NOTICE.md").write_text(NOTICE)
    counts["questions"] = write_questions(out / "labelled_data.jsonl", out / "questions.jsonl")
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", required=True, choices=["p"], help="only Track P may leave the repository")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = Path(args.out) if args.out else OUT_DIR / f"release-{args.track}"
    counts = export_release(out, OUT_DIR, f"track-{args.track}")
    print(f"release -> {out}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
