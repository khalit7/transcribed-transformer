"""One prompt, one call, one label.

`generate(case, mode, model, question=None, family=None, prefer=None)` covers
the whole 2x2:

  question given (bank)  x  mode as_is     -> judge the call as it stands
  question given (bank)  x  mode injected  -> write 1-4 turns so a target answer holds
  question None (written) x mode as_is     -> write a question for this call, then judge
  question None (written) x mode injected  -> write a question AND the turns that make
                                               its target answer hold

The prompt is one template with two optional blocks (write the question?
write turns?). For injection the generator picks a target answer whose grading
rule can be made true by ADDING turns (never an absence), preferring `prefer`
when given; the turns are inserted at the same position in every variant
(line alignment kept), channel-noised in messy variants, and the evidence is
the inserted lines. For as-is judgement the evidence is whatever lines the
model cites, clipped to range.

Returns (case, question, label, cost_usd). The case is the modified call when
turns were injected.
"""

import json

from src.synthesis.cases import noise_line, stable_seed
from src.synthesis.llm import ask_json
from src.synthesis.schema import Case, Label, Mode, Question, Variant

STYLE = ("Match the register of the surrounding conversation: spontaneous speech, natural "
         "hesitations and fillers where the rest of the call has them, no scripted or written tone, "
         "no stage directions. Use only the speaker tags that already appear. Never mention the "
         "question or the assessment.")

def numbered(case: Case) -> str:
    return "\n".join(f"{i + 1}: {ln}" for i, ln in enumerate(case.transcript.lines("clean")))


def prompt(case: Case, mode: Mode, question: Question | None, family: str | None, prefer: str | None) -> str:
    roles = ", ".join(f"{t} is the {r}" for t, r in case.transcript.speaker_roles.items())
    parts = [f"You are creating quality-assurance evaluation data from a call transcript. One turn per line, "
             f"numbered. Speaker roles (hidden from the evaluated model, known to you): {roles}.",
             f"TRANSCRIPT\n{numbered(case)}"]
    out = {}
    if question is not None:
        opts = "\n".join(f"- {o.value}: {o.criteria}" for o in question.options)
        ctx = f"\nContext: {question.description}" if question.description else ""
        parts.append(f"QUESTION: {question.text}{ctx}\nAnswer options and grading rules:\n{opts}")
    else:
        parts.append(f'Write ONE new quality-assurance question of the "{family}" family that fits THIS conversation, '
                     'with explicit grading rules. Options must include "pass" and "fail"; add "partial_pass" and/or '
                     '"NA" only if they are meaningful for the question.')
        out["question"] = '{"text": "...", "description": "...", "options": [{"value": "pass", "criteria": "..."}, {"value": "fail", "criteria": "..."}]}'
    if mode == "as_is":
        parts.append("Decide the answer strictly by the rules. Cite every line that supports the answer "
                     "(empty list if nothing in the transcript relates to the question).")
        out["answer"] = '"<one of the option values>"'
        out["evidence"] = "[<line numbers>]"
    else:
        pref = f' Prefer "{prefer}" if it can be made true this way.' if prefer else ""
        parts.append("Choose a target answer whose rule can be made true by ADDING 1 to 4 new consecutive turns to the "
                     "transcript (never an answer that describes an absence; the transcript must not already satisfy "
                     f"it).{pref} Write those turns so that the target rule holds and NO other option's rule is met. "
                     f"Choose a natural insertion point. {STYLE}")
        out["answer"] = '"<the target option value>"'
        out["insert_after_line"] = "<int, 0 to insert at the start>"
        out["turns"] = '[{"speaker": "SPEAKER_xx", "text": "..."}]'
    out["summary"] = '"<one or two sentences, past tense, explaining why the answer holds, referring to what was said>"'
    schema = "{" + ", ".join(f'"{k}": {v}' for k, v in out.items()) + "}"
    parts.append(f"Respond with a single JSON object and nothing else:\n{schema}")
    return "\n\n".join(parts)


def insert_turns(case: Case, turns: list[dict], after: int, seed_key: str) -> tuple[Case, list[int]]:
    """Insert turns after line `after` in every variant; returns the new case and the 1-based inserted lines."""
    if not 1 <= len(turns) <= 4:
        raise ValueError("injection must add 1-4 turns")
    tags = set(case.transcript.speaker_roles)
    for t in turns:
        if t["speaker"] not in tags or not str(t["text"]).strip():
            raise ValueError(f"bad injected turn: {t}")
    k = max(0, min(int(after), case.transcript.n_lines))
    variants = []
    for v in case.transcript.variants:
        if v.kind == "clean":
            add = [f"{t['speaker']}: {t['text'].strip()}" for t in turns]
        else:
            add = [f"{t['speaker']}: {noise_line(t['text'], stable_seed(seed_key, i))}" for i, t in enumerate(turns)]
        variants.append(Variant(kind=v.kind, origin=v.origin, lines=v.lines[:k] + add + v.lines[k:]))
    new = case.model_copy(update={"transcript": case.transcript.model_copy(update={"variants": variants})})
    return new, list(range(k + 1, k + 1 + len(turns)))


def generate(case: Case, mode: Mode, model: str, question: Question | None = None,
             family: str | None = None, prefer: str | None = None) -> tuple[Case, Question, Label, float]:
    if question is None and family is None:
        raise ValueError("either a question or a family to write one for is required")
    resp, cost = ask_json(prompt(case, mode, question, family, prefer), model)

    if question is None:
        qd = resp["question"]
        question = Question(id=f"written-{case.id}-{mode}", source="written", family=family, text=qd["text"],
                            description=qd.get("description", ""), options=qd["options"])
    answer = resp["answer"]
    if answer not in question.values:
        raise ValueError(f"answer {answer!r} not in {question.values}")

    if mode == "injected":
        case, evidence = insert_turns(case, resp["turns"], resp["insert_after_line"], f"{case.id}|{question.id}")
    else:
        evidence = sorted({int(e) for e in resp.get("evidence", []) if 1 <= int(e) <= case.transcript.n_lines})
    label = Label(answer=answer, evidence=evidence, summary=str(resp.get("summary", "")).strip())
    return case, question, label, cost


def dumps(obj) -> str:
    return json.dumps(obj.model_dump(), ensure_ascii=False)
