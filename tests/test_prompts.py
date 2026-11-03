"""The prompt templates live as files (data/labelled_data*/prompts/); these tests pin what the code renders from them."""

from pathlib import Path

from src import prompts
from src.synthesis import identify_speakers, label
from src.synthesis.schema import Question
from src.synthesis_v2 import (
    agent_label,
    cases,
    identify_hosts,
    label_episodes,
    question_bank,
)
from src.train import data

V1, V2 = prompts.V1, prompts.V2
GOLDEN = Path(__file__).parent / "fixtures" / "prompts"  # renderings captured from the pre-refactor code (2026-10-31)


def _q(family: str, tags: tuple[str, ...] = ()) -> Question:
    opts = ([{"value": "pass", "criteria": "c1"}, {"value": "fail", "criteria": "c2"}] if family == "general_qa"
            else [{"value": "yes", "criteria": "c1"}, {"value": "no", "criteria": "c2"}])
    return Question.model_validate({"id": f"x-{family}", "family": family, "text": "Is it so?", "description": "Some context.",
                                    "options": opts, "tags": list(tags), "dataset_allow_list": ["sporc"]})


def test_every_template_file_is_named_in_its_readme_and_has_one_trailing_newline():
    for folder in (V1, V2):
        readme = (folder / "README.md").read_text()
        files = sorted(p for p in folder.iterdir() if p.name != "README.md")
        assert files, folder
        for p in files:
            assert f"`{p.name}`" in readme, (p, "not documented in the folder's README")
            raw = p.read_bytes()
            assert raw.endswith(b"\n") and not raw.endswith(b"\n\n"), (p, "must end with exactly one newline")
            assert b"$$" not in raw, p  # no literal dollar signs are needed in any template


def test_loader_strips_one_newline_and_renders_placeholders(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("Hello $name\n{\"k\": [$values]}\n")
    assert prompts.text(p) == 'Hello $name\n{"k": [$values]}'
    assert prompts.render(p, name="x", values=[1, 2]) == 'Hello x\n{"k": [[1, 2]]}'
    try:
        prompts.render(p, name="x")
    except KeyError:
        pass
    else:
        raise AssertionError("a missing placeholder rendered")
    d1 = prompts.digest(p)
    p2 = tmp_path / "t2.txt"
    p2.write_text("changed\n")
    assert prompts.digest(p2) != d1 and prompts.digest(p) == d1
    assert prompts.render(tmp_path / "t.txt", name="y", values=1).startswith("Hello y")  # positional-only first argument


def test_renderings_equal_the_pre_refactor_strings():
    """Every prompt the code renders equals the string the code produced before the templates became files."""
    lines = [f"SPEAKER_0{i % 2}: words number {i}" for i in range(6)]
    got = {f"train_{fam}": data.task_prompt(lines, ["SPEAKER_00", "SPEAKER_01"], _q(fam)) for fam in ("general_qa", "transfer", "podcast")}
    got["train_general_qa_tags"] = data.task_prompt(lines, ["agent", "customer"], _q("general_qa", ("t1", "t2")))
    got["label_v1"] = label.prompt(lines, ["agent", "customer"], _q("general_qa", ("t1",)))
    got["label_v1_notags"] = label.prompt(lines, ["agent", "customer"], _q("general_qa"))
    turns = [(f"SPEAKER_0{i % 3}", f"turn {i}") for i in range(300)]
    got["identify_v1"] = identify_speakers.prompt(turns)
    turns2 = list(turns)
    turns2[150] = ("SPEAKER_05", "caller")
    got["identify_v2"] = identify_hosts.prompt(turns2)
    got["identify_v2_short"] = identify_hosts.prompt(turns[:50])
    text = "\n".join(f"SPEAKER_0{i % 2}: this is turn number {i} about Nashville and the Grand Ole Opry" for i in range(12))
    c = cases.build("ep1", text, ["SPEAKER_00"], {"stratum": "2-4k"}, guests=["SPEAKER_01"])
    qs = question_bank.ROUNDS_1_TO_3[:3]  # fixed questions, so the golden rendering does not move with the bank
    got["label_episodes_v2"] = label_episodes.prompt(c, qs)
    got["vocab"] = question_bank.VOCAB
    got["task_v2"] = agent_label.TASK
    for name, rendered in got.items():
        assert rendered == (GOLDEN / f"{name}.txt").read_text(), name
    assert sorted(p.stem for p in GOLDEN.glob("*.txt")) == sorted(got)  # every golden file is checked


def test_judge_and_probe_prompts_render_from_their_templates():
    from src.synthesis import grade_labels
    from src.synthesis.schema import Generation, Label, LabelledRecord
    q = _q("general_qa", ("t1",))
    text = "\n".join(f"SPEAKER_0{i % 2}: this is turn number {i} about Nashville and the Grand Ole Opry" for i in range(12))
    c = cases.build("ep1", text, ["SPEAKER_00"], {})
    g = Generation(name="t:m", labelled_variant="clean", cost_usd=0.0, timestamp="t")
    rec = LabelledRecord(id="ep1::q", dataset="sporc", source_id="s", track="track-nc", question=q, transcript=c.transcript,
                         label=Label(answer="pass", evidence=[1], summary="s", tags=["t1"]), generation_info=g)
    p = grade_labels.prompt(rec, [("A", rec), ("B", rec)])
    assert p.startswith("You are the judge of automatic labels for a conversation-transcript quality-assurance task. One turn per line, "
                        "numbered; each line starts with the speaker's role as recorded by the source (host_1, SPEAKER_01).\n\nTRANSCRIPT\n1: host_1:")
    assert ", and tags:\n\nLABEL A: answer=pass, evidence lines=[1], tags=['t1']\n  summary: s\nLABEL B:" in p
    assert ", then list the tags that apply. Then grade every label" in p
    assert "- tags: 2 if the tags match the characteristics actually disclosed, 1 if partly, 0 if wrong or missing.\n" in p
    assert p.endswith('"grades": {"A": {"answer_correct": <bool>, "evidence": <0-2>, "summary": <0-2>, "tags": <0-2 or null>, '
                      '"reason": "<one sentence>"}, "B": {"answer_correct": <bool>, "evidence": <0-2>, "summary": <0-2>, '
                      '"tags": <0-2 or null>, "reason": "<one sentence>"}}}')
    p2 = grade_labels.prompt(rec.model_copy(update={"question": _q("general_qa")}), [("A", rec)])
    assert "its reasoning:\n\nLABEL A" in p2 and "- tags: null (this question takes no tags).\n" in p2
    probe = prompts.render(V1 / "probe_bank.txt", speakers="agent, customer", transcript="1: agent: hi", questions="q1: x Options: pass = a")
    assert probe == ("Transcript, one turn per line, each prefixed by the speaker's role as recorded by the source (agent, customer).\n\n"
                     "TRANSCRIPT\n1: agent: hi\n\nAnswer EVERY question below strictly by its option criteria, for this call as it stands.\n\n"
                     'q1: x Options: pass = a\n\nRespond with one JSON object mapping each question id to its answer value, e.g. '
                     '{"gen-01-greeting": "pass", ...}. Nothing else.')


def test_the_two_gate_script_copies_are_identical():
    a = Path(".claude/skills/pre-run-review/scripts/gate.py").read_text()
    assert a == Path(".agents/skills/pre-run-review/scripts/gate.py").read_text()
    assert '"data/labelled_data/prompts", "data/labelled_data_v2/prompts"' in a


def test_training_prompt_by_family_matches_the_fixed_wording():
    lines = [f"SPEAKER_0{i % 2}: words number {i}" for i in range(3)]
    pod = data.task_prompt(lines, ["host_1", "guest_1"], _q("podcast"))
    assert pod.startswith("You answer questions about the transcript of a podcast episode. One turn per line, numbered. Each line starts "
                          "with a speaker label: host_1, host_2, ... for the hosts, guest_1, guest_2, ... for the guests, or the "
                          "source's diarisation tag for any other voice (host_1, guest_1).\n\nTRANSCRIPT\n1. SPEAKER_00:words number 0\n")
    assert pod.endswith('Respond with a single JSON object and nothing else:\n{"evidence": [<line numbers>], "answer": "<one of '
                        "['yes', 'no']>\", \"summary\": \"<reasoning>\"}")
    tr = data.task_prompt(lines, ["text"], _q("transfer"))
    assert tr.startswith("You answer questions about a document. One paragraph per line, numbered. Each line starts with the fixed label text.\n")
    comp = data.task_prompt(lines, ["agent", "customer"], _q("general_qa", ("t1", "t2")))
    assert comp.startswith("You assess conversation transcripts for quality assurance. One turn per line, numbered. Each line starts with the "
                           "speaker's role as recorded by the source (agent, customer).\n")
    assert "\nThe answer must also carry tags chosen only from this list (empty list if none applies): t1; t2\n" in comp
    assert comp.endswith('"summary": "<reasoning>", "tags": ["<tag>", ...]}')
    # the three family templates differ only in their first line
    bodies = {p.name: prompts.text(p).split("\n", 1)[1] for p in (data.PROMPT_DEFAULT, *data.PROMPT_TEMPLATES.values())}
    assert len(set(bodies.values())) == 1, bodies.keys()
    assert data.prompt_template("vulnerability") == data.PROMPT_DEFAULT and data.prompt_template("podcast") == prompts.V2 / "train_task.txt"


def test_v1_labeller_and_identifier_prompts_render():
    lines = ["agent: hi", "customer: hello"]
    p = label.prompt(lines, ["agent", "customer"], _q("general_qa", ("t1",)))
    assert p.startswith("You assess conversation transcripts for quality assurance.") and "Finally give your confidence" in p
    assert p.endswith('"summary": "<reasoning>", "confidence": <0-1>, "tags": ["<tag>", ...]}')
    turns = [(f"SPEAKER_0{i % 2}", f"turn {i}") for i in range(10)]
    ip = identify_speakers.prompt(turns)
    assert "The speaker tags are SPEAKER_00, SPEAKER_01." in ip and "\"<one of ['SPEAKER_00', 'SPEAKER_01']> or null\"" in ip


def test_v2_prompts_come_from_the_files():
    assert question_bank.VOCAB == prompts.text(V2 / "definitions.txt") and question_bank.VOCAB.startswith("Definitions. A host is any speaker")
    assert agent_label.TASK == (V2 / "labelling_task.md").read_text() and agent_label.TASK.startswith("# Labelling task")
    text = "\n".join(f"SPEAKER_0{i % 2}: this is turn number {i} about Nashville and the Grand Ole Opry" for i in range(12))
    c = cases.build("ep1", text, ["SPEAKER_00"], {"stratum": "2-4k"}, guests=["SPEAKER_01"])
    qs = question_bank.BANK[:2]
    lp = label_episodes.prompt(c, qs)
    preamble = lp.split("\n\nTRANSCRIPT\n")[0]
    assert preamble == data.task_prompt(c.transcript.lines("messy"), c.transcript.speakers, qs[0]).split("\n\nTRANSCRIPT\n")[0]
    assert "Answer each of the following 2 questions" in lp and lp.endswith('"confidence": <0-1>}}')
    ihp = identify_hosts.prompt([("SPEAKER_00", "a"), ("SPEAKER_01", "b")])
    assert ihp.startswith("Below are the opening and closing turns") and '"kind": "conversation" | "scripted" | "other"' in ihp
    assert "[<zero or more of SPEAKER_00, SPEAKER_01>]" in ihp
