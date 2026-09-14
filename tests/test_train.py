"""Round-trip and plan tests for src/train (no GPU)."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.synthesis.schema import Label, Question
from src.train.data import SEP, Example, plan_epoch, target_text, task_prompt
from src.train.evaluate import macro_f1, parse, score_pair

Q = Question(id="gen-01-greeting", family="general_qa", text="Did the agent greet the customer?", dataset_allow_list=["apptek"],
             options=[{"value": "pass", "criteria": "A greeting is given."}, {"value": "fail", "criteria": "No greeting."},
                      {"value": "NA", "criteria": "Not applicable."}])
REC = {"id": "call-1::gen-01-greeting", "dataset": "apptek", "track": "track-p", "cell": "seen_q",
       "question": Q.model_dump(), "label": {"answer": "pass", "evidence": [1], "summary": "Greeted.", "tags": []},
       "transcript": {"variants": [{"kind": "clean", "origin": "x", "lines": ["agent: hello", "customer: hi", "agent: bye"]}],
                      "speakers": ["agent", "customer"], "role_source": "x"}}


def test_prompt_and_target():
    p = task_prompt(REC["transcript"]["variants"][0]["lines"], ["agent", "customer"], Q)
    assert "1: agent: hello\n2: customer: hi\n3: agent: bye" in p
    assert "confidence" not in p and '"tags"' not in p
    t = target_text(Label(answer="pass", evidence=[1], summary="Greeted."), Q)
    assert json.loads(t) == {"evidence": [1], "answer": "pass", "summary": "Greeted."}
    assert SEP == "\n\n"


def test_plan_epoch_covers_every_example_once_and_respects_budget():
    ex = [Example(id=str(i), variant="clean", input_ids=np.arange(n, dtype=np.int32), n_prompt=n - 3)
          for i, n in enumerate([100, 5000, 300, 9000, 250, 120, 7000, 4000, 20000, 600, 33, 800])]
    steps = plan_epoch(ex, batch_sequences=4, micro_tokens=8192, world_size=2, seed=0, epoch=0)
    assert len(steps) == 3
    seen = [i for s in steps for rank in s.micro for m in rank for i in m]
    assert sorted(seen) == list(range(12))
    for s in steps:
        assert s.target_tokens == sum(ex[i].n_target for rank in s.micro for m in rank for i in m)
        for rank in s.micro:
            assert rank, "every rank gets at least one micro-batch"
            for m in rank:
                longest = max(len(ex[i].input_ids) for i in m)
                assert len(m) == 1 or longest * len(m) <= 8192
    assert plan_epoch(ex, 4, 8192, 2, 0, 0) == steps  # deterministic
    assert plan_epoch(ex, 4, 8192, 2, 0, 1) != steps  # reshuffled per epoch


def test_score_pair_gates_and_status():
    ok = score_pair(REC, {"variant": "clean", "text": '{"evidence": [1], "answer": "pass", "summary": "x"}'})
    assert ok["status"] == "exact" and ok["format_valid"] and ok["ev_exact"] == 1.0 and ok["ev_prec"] == 1.0
    rec = score_pair(REC, {"variant": "clean", "text": 'Sure! {"evidence": [1, 2], "answer": "Pass", "summary": "x"}'})
    assert rec["status"] == "recovered" and rec["credit"] == 0.3 and not rec["format_valid"] and rec["ev_prec"] == 0.5
    bad_ev = score_pair(REC, {"variant": "clean", "text": '{"evidence": "1, 2", "answer": "pass", "summary": "x"}'})
    assert bad_ev["status"] == "exact" and not bad_ev["ev_valid"] and bad_ev["ev_prec"] == 0.0
    out_of_range = score_pair(REC, {"variant": "clean", "text": '{"evidence": [0, 4], "answer": "pass", "summary": "x"}'})
    assert not out_of_range["ev_valid"]
    garbage = score_pair(REC, {"variant": "clean", "text": "no json here"})
    assert garbage["status"] == "invalid" and garbage["pred"] == "<invalid>" and not garbage["parsed"]
    # empty means empty
    empty_gold = {**REC, "label": {"answer": "NA", "evidence": [], "summary": "", "tags": []}}
    hit = score_pair(empty_gold, {"variant": "clean", "text": '{"evidence": [], "answer": "NA", "summary": ""}'})
    miss = score_pair(empty_gold, {"variant": "clean", "text": '{"evidence": [2], "answer": "NA", "summary": ""}'})
    assert hit["ev_exact"] == 1.0 and hit["ev_prec"] == 1.0 and miss["ev_exact"] == 0.0 and miss["ev_prec"] == 0.0


def test_macro_f1_and_parse():
    rows = [{"gold": "pass", "pred": "pass"}, {"gold": "fail", "pred": "pass"}, {"gold": "fail", "pred": "fail"}]
    assert abs(macro_f1(rows) - (2 / 3 + 2 / 3) / 2) < 1e-9
    assert parse('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse("[1, 2]") is None


@pytest.mark.skipif(not any(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3-1.7B-Base")), reason="tokenizer not cached")
def test_build_examples_masks_prompt(tmp_path):
    from transformers import AutoTokenizer

    from src.train.data import build_examples
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    split = tmp_path / "s.jsonl"
    split.write_text(json.dumps({**REC, "source_id": "x", "meta": {}, "generation_info": {"name": "m", "labelled_variant": "clean"}}) + "\n")
    ex, stats = build_examples(split, tok, ["labelled"], 32768)
    ids = ex[0].input_ids.tolist()
    assert stats["examples"] == 1 and ids[-1] == tok.eos_token_id
    assert tok.decode(ids[ex[0].n_prompt:-1]) == target_text(Label.model_validate(REC["label"]), Q)
    assert tok.decode(ids[:ex[0].n_prompt]).endswith(SEP)


def test_mntp_mask_is_deterministic_and_masks_only_real_tokens():
    import torch

    from src.train.data import mntp_mask
    ids = torch.tensor([[5, 6, 7, 8, 9, 0, 0], [5, 6, 7, 8, 9, 10, 11]])
    attn = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
    batch = {"input_ids": ids, "attention_mask": attn, "labels": ids.clone(), "prompt_len": torch.zeros(2, dtype=torch.long)}
    a = mntp_mask(batch, 0.5, mask_id=99, seed=3)
    b = mntp_mask(batch, 0.5, mask_id=99, seed=3)
    assert torch.equal(a["input_ids"], b["input_ids"]) and torch.equal(a["labels"], b["labels"])
    masked = a["input_ids"] == 99
    assert masked.sum() > 0 and not masked[:, 0].any() and not masked[0, 5:].any()
    assert torch.equal(a["labels"] != -100, masked)
    assert torch.equal(a["labels"][masked], ids[masked])
    assert a["prompt_len"].tolist() == [5, 7]  # bidirectional over every real token
    assert not torch.equal(mntp_mask(batch, 0.5, mask_id=99, seed=4)["input_ids"], a["input_ids"])
