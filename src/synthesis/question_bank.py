"""The question bank: reusable questions that apply to any call. This module is
the source of truth for them (source="bank"). data/benchmark/questions.jsonl is
derived: the bank plus every question written for a specific call, collected
from labelled_data.jsonl.

Original questions in the generic shape of regulated-industry call QA: the
vulnerability family follows the FCA's four drivers (health, life events,
resilience, capability) from the public guidance FG21/1; complaints and
general-conduct questions are ordinary call-quality practice. Every question
carries pass and fail; partial_pass and NA are optional per question.

Run: uv run python -m src.synthesis.question_bank   (regenerates questions.jsonl)
"""

import json
from pathlib import Path

from src.synthesis.schema import Question

BENCH = Path(__file__).resolve().parents[2] / "data" / "benchmark"


def q(id, family, text, options, description=""):
    return Question(
        id=id, source="bank", family=family, text=text, description=description,
        options=[{"value": v, "criteria": c} for v, c in options],
    )


QUESTIONS: list[Question] = [
    q('vul-health-01', 'vulnerability',
      'Did the customer disclose a health condition that could affect their ability to manage the matter discussed?',
      [('pass', 'The customer clearly mentions a physical or mental health condition, illness, disability or treatment affecting them.'),
       ('partial_pass', "The customer hints at a health issue without stating it clearly (e.g. 'I've not been well lately') and it is not clarified."),
       ('fail', 'No health condition is disclosed or implied anywhere in the call.'),
       ('NA', 'Not applicable: the call is not with the customer themselves.')]),
    q('cmp-raised-01', 'complaint_and_eod',
      'Did the customer make a complaint during the call?',
      [('pass', "The customer expresses dissatisfaction about the firm, its product, service or staff, whether or not they use the word 'complaint'."),
       ('partial_pass', "The customer expresses mild frustration about something incidental (e.g. hold time) without complaining about the firm's conduct or product."),
       ('fail', 'No dissatisfaction is expressed.')]),
    q('gen-dob-01', 'general_qa',
      "Was the customer's date of birth established during the call?",
      [('pass', 'The agent asks for the date of birth and the customer gives it, or the customer volunteers it.'),
       ('fail', 'The date of birth is never asked for or given.')]),
]

assert len({x.id for x in QUESTIONS}) == len(QUESTIONS), "duplicate question ids"


def all_questions(labelled_path: Path = BENCH / "labelled_data.jsonl") -> list[Question]:
    """The bank plus every distinct question embedded in the labelled data."""
    seen = {x.id for x in QUESTIONS}
    out = list(QUESTIONS)
    if labelled_path.exists():
        for line in labelled_path.open():
            qd = json.loads(line)["question"]
            if qd["id"] not in seen:
                seen.add(qd["id"])
                out.append(Question.model_validate(qd))
    return out


def write_questions(labelled_path: Path = BENCH / "labelled_data.jsonl",
                    out_path: Path = BENCH / "questions.jsonl") -> int:
    qs = all_questions(labelled_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for x in qs:
            f.write(x.model_dump_json() + "\n")
    return len(qs)


if __name__ == "__main__":
    from collections import Counter
    n = write_questions()
    print(f"wrote {n} questions ({len(QUESTIONS)} bank) -> {BENCH / 'questions.jsonl'}")
    print(Counter(x.family for x in QUESTIONS))
