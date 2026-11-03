"""LLM labelling of one (call, question): the labeller reads the transcript and answers.

`label(case, question, model)` asks for evidence FIRST, then the answer, then
the summary — the quote-then-answer order that improves citation quality in the
literature — plus optional tags (when the question has a tag vocabulary) and
the labeller's own 0-1 confidence. The same schema is enforced server-side on
Ollama and requested in the prompt for `claude -p`.

`verify()` has a second model answer blind and records agreement.
`ablate()` re-labels with the cited lines removed (necessary?) and with only the
cited lines kept (sufficient?), so an evidence key can be trusted beyond the
labeller's say-so.
"""

import json

from src import prompts
from src.synthesis.llm import ask_json
from src.synthesis.schema import Ablation, Case, Label, Question, Verification


def numbered(lines: list[str]) -> str:
    return "\n".join(f"{i + 1}: {ln}" for i, ln in enumerate(lines))


def schema_for(q: Question) -> dict:
    props = {
        "evidence": {"type": "array", "items": {"type": "integer", "minimum": 1}},
        "answer": {"type": "string", "enum": q.values},
        "summary": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    required = ["evidence", "answer", "summary", "confidence"]
    if q.tags:
        props["tags"] = {"type": "array", "items": {"type": "string", "enum": q.tags}}
        required.append("tags")
    return {"type": "object", "properties": props, "required": required}


PROMPT = prompts.V1 / "label.txt"  # the v1 labeller's prompt, one question per call


def prompt(lines: list[str], speakers: list[str], q: Question) -> str:
    opts = "\n".join(f"- {o.value}: {o.criteria}" for o in q.options)
    ctx = f"\nContext: {q.description}" if q.description else ""
    tag_txt = ("\nThe answer must also carry tags chosen only from this list (empty list if none applies): "
               + "; ".join(q.tags)) if q.tags else ""
    tag_field = ', "tags": ["<tag>", ...]' if q.tags else ""
    return prompts.render(PROMPT, speakers=", ".join(speakers), transcript=numbered(lines), question=q.text, context=ctx,
                          options=opts, tags_rule=tag_txt, values=q.values, tag_field=tag_field)


def _parse(resp: dict, q: Question, n_lines: int) -> Label:
    answer = resp["answer"]
    if answer not in q.values:
        raise ValueError(f"answer {answer!r} not in {q.values}")
    evidence = sorted({int(e) for e in resp.get("evidence", []) if 1 <= int(e) <= n_lines})
    tags = [t for t in resp.get("tags", []) if t in q.tags] if q.tags else []
    conf = resp.get("confidence")
    conf = min(1.0, max(0.0, float(conf))) if isinstance(conf, (int, float)) else None
    return Label(answer=answer, evidence=evidence, summary=str(resp.get("summary", "")).strip(),
                 tags=tags, confidence=conf)


def label(case: Case, q: Question, model: str, variant: str = "clean", route: str | int | None = None) -> tuple[Label, float]:
    """`route` picks the Ollama server when several are configured (default: by call id; a caller that
    schedules calls can pass a call index so consecutive calls alternate servers)."""
    lines = case.transcript.lines(variant)
    resp, cost = ask_json(prompt(lines, case.transcript.speakers, q), model, schema_for(q),
                          route=case.id if route is None else route)
    return _parse(resp, q, len(lines)), cost


def verify(case: Case, q: Question, primary: Label, model: str, variant: str = "clean") -> tuple[Verification, float]:
    second, cost = label(case, q, model, variant)
    return Verification(model=model, answer=second.answer, evidence=second.evidence, tags=second.tags,
                        agrees=second.answer == primary.answer), cost


def ablate(case: Case, q: Question, primary: Label, model: str, variant: str = "clean") -> tuple[Ablation, float]:
    """Necessity: remove the cited lines (keep numbering by blanking them); the answer should change.
    Sufficiency: keep only the cited lines (others blanked); the answer should hold."""
    if not primary.evidence:
        return Ablation(model=model, necessary=None, sufficient=None), 0.0
    lines = case.transcript.lines(variant)
    cited = set(primary.evidence)
    speakers = case.transcript.speakers
    removed = [ln if (i + 1) not in cited else ln.split(": ", 1)[0] + ": [removed]" for i, ln in enumerate(lines)]
    only = [ln if (i + 1) in cited else ln.split(": ", 1)[0] + ": [removed]" for i, ln in enumerate(lines)]
    total = 0.0
    r1, c1 = ask_json(prompt(removed, speakers, q), model, schema_for(q), route=case.id); total += c1
    r2, c2 = ask_json(prompt(only, speakers, q), model, schema_for(q), route=case.id); total += c2
    return Ablation(model=model, necessary=r1.get("answer") != primary.answer,
                    sufficient=r2.get("answer") == primary.answer), total


def dumps(obj) -> str:
    return json.dumps(obj.model_dump(), ensure_ascii=False)
