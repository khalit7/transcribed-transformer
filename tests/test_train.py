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
    rec = score_pair(REC, {"variant": "clean", "text": ' {"evidence": [1, 2], "answer": "Pass", "summary": "x"}\n'})
    assert rec["status"] == "recovered" and rec["credit"] == 0.3 and not rec["format_valid"] and rec["ev_prec"] == 0.5
    assert rec["pred"] == "<invalid>" and rec["pred_raw"] == "pass"  # partial credit only: never a correct prediction
    # strict format: the whole output is one JSON object, with a summary string
    wrapped = score_pair(REC, {"variant": "clean", "text": 'Sure! {"evidence": [1], "answer": "pass", "summary": "x"}'})
    assert wrapped["status"] == "invalid" and not wrapped["parsed"] and wrapped["ev_prec"] == 0.0
    no_summary = score_pair(REC, {"variant": "clean", "text": '{"evidence": [1], "answer": "pass"}'})
    assert no_summary["status"] == "exact" and not no_summary["format_valid"]
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
    # the classes are the questions' option values: predicting an option that never occurs as gold is a false positive
    # for that class, and an option nobody predicts or holds as gold is left out
    opts = [{**r, "options": ["pass", "fail", "NA"]} for r in rows] + [{"gold": "pass", "pred": "NA", "options": ["pass", "fail", "NA"]}]
    assert abs(macro_f1(opts) - (2 / 4 + 2 / 3 + 0.0) / 3) < 1e-9
    assert abs(macro_f1([{**r, "options": ["pass", "fail", "NA"]} for r in rows]) - (2 / 3 + 2 / 3) / 2) < 1e-9
    assert parse('```json\n{"a": 1}\n```') is None  # strict: one JSON object and nothing else
    assert parse(' {"a": 1}\n') == {"a": 1}
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


def test_build_examples_opens_with_the_tokenizer_start_token(tmp_path):
    """A Gemma tokenizer adds <bos>; the prompt must carry it (T5Gemma 2 and Gemma 3 were trained with it)."""
    import pytest
    from transformers import AutoTokenizer

    from src.train.data import build_examples
    try:
        tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt", local_files_only=True)
    except OSError:  # gated model not on this machine
        pytest.skip("gemma-3-1b-pt tokenizer not cached")
    split = tmp_path / "s.jsonl"
    split.write_text(json.dumps({**REC, "source_id": "x", "meta": {}, "generation_info": {"name": "m", "labelled_variant": "clean"}}) + "\n")
    ex, _ = build_examples(split, tok, ["labelled"], 32768)
    ids = ex[0].input_ids.tolist()
    assert ids[0] == tok.bos_token_id and ids.count(tok.bos_token_id) == 1 and ids[-1] == tok.eos_token_id
    assert tok.decode(ids[ex[0].n_prompt:-1]) == target_text(Label.model_validate(REC["label"]), Q)


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


def test_collate_encdec_finetune_and_seq2seq():
    import torch

    from src.train.data import collate_encdec
    ex = [Example(id="a", variant="c", input_ids=np.arange(10, 30, dtype=np.int32), n_prompt=15),
          Example(id="b", variant="c", input_ids=np.arange(100, 108, dtype=np.int32), n_prompt=5)]
    b = collate_encdec(ex, [0, 1], pad_id=0, start_id=999)
    assert b["enc_ids"].shape == (2, 15) and b["dec_ids"].shape == (2, 5)
    assert b["enc_ids"][1, 5:].eq(0).all() and b["enc_mask"][1].tolist() == [1] * 5 + [0] * 10
    assert b["dec_ids"][0].tolist() == [999, 25, 26, 27, 28] and b["labels"][0].tolist() == [25, 26, 27, 28, 29]
    assert b["dec_ids"][1].tolist() == [999, 105, 106, 0, 0] and b["labels"][1].tolist() == [105, 106, 107, -100, -100]
    s = collate_encdec(ex, [0], pad_id=0, start_id=999, seed=1, seq2seq=True, max_target=3, cut=(0.5, 0.5))
    assert s["enc_ids"].shape[1] == 10 and s["labels"][0].tolist() == [20, 21, 22] and s["dec_ids"][0].tolist() == [999, 20, 21]
    assert torch.equal(collate_encdec(ex, [0], 0, 999, seed=2, seq2seq=True)["labels"], collate_encdec(ex, [0], 0, 999, seed=2, seq2seq=True)["labels"])


def test_shard_units_lists_blocks_and_keeps_towers_whole():
    """Every transformer block is a unit; a frozen image tower is one unit and its inner layers are not listed."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from src.train.train import shard_units
    model = Qwen3ForCausalLM(Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=2,
                                         num_key_value_heads=1, head_dim=16, vocab_size=100))
    tower = torch.nn.Module()
    tower.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)])
    model.vision_tower = tower
    blocks, towers = shard_units(model)
    assert len(blocks) == 3 and towers == [tower]


def test_api_baseline_request_caches_the_transcript_block():
    """The transcript-bearing prefix is the cached block; the question follows uncached; cost uses batch prices."""
    from src.train.api_baseline import SPLIT_MARK, cost_usd, request_params
    prompt = "instructions\n\nTRANSCRIPT\n1: A: hi\n" + SPLIT_MARK + "was it polite?\nAnswer options..."
    p = request_params(prompt, "claude-sonnet-5", cache=True)
    blocks = p["messages"][0]["content"]
    assert len(blocks) == 2 and blocks[0]["cache_control"] == {"type": "ephemeral"} and "cache_control" not in blocks[1]
    assert blocks[0]["text"] + blocks[1]["text"] == prompt and blocks[1]["text"].startswith(SPLIT_MARK)
    assert p["temperature"] == 0.0 and p["max_tokens"] == 512
    usage = {"input_tokens": 1_000_000, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1_000_000, "output_tokens": 100_000}
    assert abs(cost_usd("claude-sonnet-5", usage) - 0.5 * (2.0 + 0.2 + 1.0)) < 1e-9


def test_collate_encdec_dec_prompt_feeds_the_prompt_to_the_decoder_with_labels_on_the_target_only():
    import numpy as np
    import torch

    from src.train.data import Example, collate_encdec
    ex = [Example(id="a", variant="v", input_ids=np.array([11, 12, 13, 21, 22, 1], dtype=np.int32), n_prompt=3),
          Example(id="b", variant="v", input_ids=np.array([31, 41, 1], dtype=np.int32), n_prompt=1)]
    b = collate_encdec(ex, [0, 1], pad_id=0, start_id=2, dec_prompt=True)
    assert b["dec_ids"][0].tolist() == [2, 11, 12, 13, 21, 22] and b["labels"][0].tolist() == [-100, -100, -100, 21, 22, 1]
    assert b["dec_ids"][1].tolist() == [2, 31, 41, 0, 0, 0] and b["labels"][1].tolist() == [-100, 41, 1, -100, -100, -100]
    assert b["dec_mask"].sum(1).tolist() == [6, 3] and torch.equal(b["enc_ids"][1, :1], torch.tensor([31]))
    plain = collate_encdec(ex, [0, 1], pad_id=0, start_id=2)
    assert plain["dec_ids"][0].tolist() == [2, 21, 22] and plain["labels"][0].tolist() == [21, 22, 1]


def test_collate_encdec_without_a_start_token_feeds_the_prompt_as_is():
    import numpy as np

    from src.train.data import Example, collate_encdec
    ex = [Example(id="a", variant="v", input_ids=np.array([2, 12, 13, 21, 22, 1], dtype=np.int32), n_prompt=3),
          Example(id="b", variant="v", input_ids=np.array([2, 41, 1], dtype=np.int32), n_prompt=1)]
    b = collate_encdec(ex, [0, 1], pad_id=0, start_id=2, dec_prompt=True, dec_start=False)
    # the decoder side is the prompt (which already opens with the tokenizer's start token) then the target, as in E1
    assert b["dec_ids"][0].tolist() == [2, 12, 13, 21, 22] and b["labels"][0].tolist() == [-100, -100, 21, 22, 1]
    assert b["dec_ids"][1].tolist() == [2, 41, 0, 0, 0] and b["labels"][1].tolist() == [41, 1, -100, -100, -100]
    assert b["dec_mask"].sum(1).tolist() == [5, 2] and b["enc_ids"][1].tolist()[:1] == [2] and b["enc_mask"][1].sum() == 1


def _tiny_hybrid():
    import torch
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    from transformers.models.t5gemma2.configuration_t5gemma2 import T5Gemma2TextConfig
    from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2TextEncoder

    from src.train.hybrid import Hybrid
    torch.manual_seed(0)
    dcfg = Gemma3TextConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=16, sliding_window=4, layer_types=["sliding_attention", "full_attention", "sliding_attention"],
                            query_pre_attn_scalar=16, attn_implementation="sdpa")
    dec = Gemma3ForCausalLM(dcfg).eval()
    ecfg = T5Gemma2TextConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=2,
                              num_key_value_heads=1, head_dim=16, sliding_window=4, layer_types=["sliding_attention", "full_attention"],
                              query_pre_attn_scalar=16, attn_implementation="sdpa", dropout_rate=0.0, attention_dropout=0.0)
    enc = T5Gemma2TextEncoder(ecfg).eval()
    plain = Gemma3ForCausalLM(dcfg).eval()
    plain.load_state_dict(dec.state_dict())
    return Hybrid(dec, enc, "dec", "enc", "sdpa", gradient_checkpointing=True).eval(), plain


def test_hybrid_is_the_plain_decoder_at_initialisation_and_reads_the_encoder_once_trained():
    import torch
    hybrid, plain = _tiny_hybrid()
    enc_ids = torch.tensor([[2, 5, 6, 7, 8], [2, 9, 10, 0, 0]]); enc_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    dec_ids = torch.tensor([[2, 5, 6, 7, 8, 11, 12], [2, 9, 10, 11, 12, 0, 0]]); dec_mask = torch.tensor([[1] * 7, [1] * 5 + [0, 0]])
    with torch.no_grad():
        h = hybrid(enc_ids, enc_mask, dec_ids, dec_mask)
        ref = plain.model(input_ids=dec_ids, attention_mask=dec_mask).last_hidden_state
    assert torch.allclose(h[dec_mask.bool()], ref[dec_mask.bool()], atol=1e-5)  # closed gate (post-norm weight -1): E1's model exactly
    assert all(float(layer.post_cross_layernorm.weight.abs().max()) == 1.0 for layer in hybrid.layers)
    for layer in hybrid.layers:  # every layer's cross-attention is live once its gate opens
        torch.nn.init.constant_(layer.post_cross_layernorm.weight, -0.5)
    hybrid.collect_cross_stats(True)
    with torch.no_grad():
        h2 = hybrid(enc_ids, enc_mask, dec_ids, dec_mask)
    hybrid.collect_cross_stats(False)
    assert not torch.allclose(h2[dec_mask.bool()], ref[dec_mask.bool()], atol=1e-3)
    stats = hybrid.cross_stats()
    assert stats["cross_ratio_mean"] > 0 and stats["cross_ratio_max"] >= stats["cross_ratio_mean"]
    # padding on the encoder side is never attended: masking the padded keys' content changes nothing
    enc_ids2 = enc_ids.clone(); enc_ids2[1, 3:] = 33
    with torch.no_grad():
        h3 = hybrid(enc_ids2, enc_mask, dec_ids, dec_mask)
    assert torch.allclose(h3[dec_mask.bool()], h2[dec_mask.bool()], atol=1e-5)
    # training mode with checkpointing gives the same values and gradients reach the new sublayer and the encoder
    hybrid.train()
    h4 = hybrid(enc_ids, enc_mask, dec_ids, dec_mask)
    assert torch.allclose(h4[dec_mask.bool()], h2[dec_mask.bool()], atol=1e-5)
    h4[dec_mask.bool()].pow(2).sum().backward()
    assert hybrid.layers[0].cross.q_proj.weight.grad is not None and hybrid.layers[0].cross.q_proj.weight.grad.abs().sum() > 0
    assert hybrid.encoder.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0


def test_hybrid_cached_decoding_matches_the_full_forward():
    import torch
    from transformers import DynamicCache
    hybrid, _ = _tiny_hybrid()
    for layer in hybrid.layers:
        torch.nn.init.constant_(layer.post_cross_layernorm.weight, -0.5)
    enc_ids = torch.tensor([[2, 5, 6, 7, 8], [2, 9, 10, 0, 0]]); enc_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    # full forward over prompt + two label tokens, right-padded
    dec_ids = torch.tensor([[2, 5, 6, 7, 8, 11, 12], [2, 9, 10, 11, 12, 0, 0]]); dec_mask = torch.tensor([[1] * 7, [1] * 5 + [0, 0]])
    with torch.no_grad():
        full = hybrid(enc_ids, enc_mask, dec_ids, dec_mask)
        # prefill the left-padded prompts with explicit positions, then two cached steps (the generation loop's shape)
        pre = torch.tensor([[2, 5, 6, 7, 8], [0, 0, 2, 9, 10]]); pm = torch.tensor([[1] * 5, [0, 0, 1, 1, 1]])
        pos = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 0, 1, 2]])
        enc = hybrid.encode(enc_ids, enc_mask)
        hybrid.set_context(enc, enc_mask, cache_kv=True)
        cache = DynamicCache(config=hybrid.config)
        h = hybrid.decode(pre, pm, pos, cache)
        outs = [h[:, -1]]
        npos = pos[:, -1] + 1
        for t in (11, 12):
            pm = torch.cat([pm, torch.ones(2, 1, dtype=torch.long)], 1)
            h = hybrid.decode(torch.full((2, 1), t), pm, npos[:, None], cache)
            outs.append(h[:, -1]); npos = npos + 1
        hybrid.set_context(None, None)
    assert torch.allclose(outs[0][0], full[0, 4], atol=1e-4) and torch.allclose(outs[1][0], full[0, 5], atol=1e-4) and torch.allclose(outs[2][0], full[0, 6], atol=1e-4)
    assert torch.allclose(outs[0][1], full[1, 2], atol=1e-4) and torch.allclose(outs[1][1], full[1, 3], atol=1e-4) and torch.allclose(outs[2][1], full[1, 4], atol=1e-4)


def test_param_groups_apply_lr_multipliers_by_name():
    import torch

    from src.train.config import DataConfig, OptimConfig, TrainConfig
    from src.train.train import MasterOptimizer
    cfg = TrainConfig(experiment="e4", name="t", data=DataConfig(), optim=OptimConfig(lr=1e-3, optimizer="adamw", lr_mult={"cross": 10}))
    m = torch.nn.ModuleDict({"cross": torch.nn.Linear(4, 4), "plain": torch.nn.Linear(4, 4), "norm": torch.nn.LayerNorm(4)})
    opt = MasterOptimizer(m, cfg)
    opt.set_lr(1e-3)
    assert {g["lr_mult"] for g in opt.param_groups} == {1.0, 10.0}
    assert all(abs(g["lr"] - 1e-3 * g["lr_mult"]) < 1e-12 for g in opt.param_groups)
    assert sum(p.numel() for g in opt.param_groups if g["lr_mult"] == 10.0 for p in g["params"]) == 20  # cross weight (decay) + bias (no decay)
    assert {g["weight_decay"] for g in opt.param_groups} == {0.0, cfg.optim.weight_decay} and len(opt.param_groups) == 4


def test_freeze_except_keeps_only_the_named_parameters_trainable():
    import torch

    from src.train.train import freeze_except
    m = torch.nn.ModuleDict({"encoder": torch.nn.Linear(4, 4, bias=False), "decoder": torch.nn.ModuleDict({"cross": torch.nn.Linear(4, 4, bias=False), "mlp": torch.nn.Linear(4, 4, bias=False)})})
    n_on, n_off = freeze_except(m, ["encoder", "cross"])
    assert n_on == 32 and n_off == 16
    assert m["encoder"].weight.requires_grad and m["decoder"]["cross"].weight.requires_grad and not m["decoder"]["mlp"].weight.requires_grad


def test_hybrid_gate_gradient_is_well_scaled_and_the_projection_gets_none_while_closed():
    import torch
    hybrid, _ = _tiny_hybrid()
    hybrid.train()
    enc_ids = torch.tensor([[2, 5, 6, 7, 8]]); enc_mask = torch.ones(1, 5, dtype=torch.long)
    dec_ids = torch.tensor([[2, 5, 6, 7, 8, 11, 12]]); dec_mask = torch.ones(1, 7, dtype=torch.long)
    h = hybrid(enc_ids, enc_mask, dec_ids, dec_mask)
    h.pow(2).mean().backward()
    g_gate = hybrid.layers[0].post_cross_layernorm.weight.grad
    g_base = hybrid.layers[0].inner.mlp.down_proj.weight.grad
    assert g_gate is not None and torch.isfinite(g_gate).all()
    # the gate's gradient is of the same order as the base model's, not the 1/sqrt(eps) blow-up of a norm at zero input
    assert g_gate.abs().max() < 100 * g_base.abs().max()
    assert hybrid.layers[0].cross.o_proj.weight.grad.abs().sum() == 0  # closed gate: nothing reaches the projection or the encoder
    g_enc = hybrid.encoder.layers[0].self_attn.q_proj.weight.grad
    assert g_enc is None or g_enc.abs().sum() == 0


def test_hybrid_flash_varlen_cross_attention_matches_sdpa():
    import torch
    pytest.importorskip("flash_attn")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    try:
        free = torch.cuda.mem_get_info()[0]
    except RuntimeError:  # a full card can refuse even the context (torch.AcceleratorError is a RuntimeError)
        free = 0
    if free < 2 * 2**30:
        pytest.skip("needs 2 GiB free on the GPU (a training run holds it)")
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    from src.train.hybrid import CrossAttention
    cfg = Gemma3TextConfig(vocab_size=64, hidden_size=64, intermediate_size=96, num_hidden_layers=1, num_attention_heads=4,
                           num_key_value_heads=1, head_dim=32, sliding_window=4, layer_types=["full_attention"], query_pre_attn_scalar=32)
    torch.manual_seed(0)
    attn = Gemma3ForCausalLM(cfg).model.layers[0].self_attn
    cross = CrossAttention(attn, cfg).to("cuda", torch.bfloat16)
    hidden = torch.randn(3, 11, 64, device="cuda", dtype=torch.bfloat16); enc = torch.randn(3, 9, 64, device="cuda", dtype=torch.bfloat16)
    q_mask = torch.tensor([[1] * 11, [1] * 7 + [0] * 4, [1] * 3 + [0] * 8], device="cuda"); enc_mask = torch.tensor([[1] * 9, [1] * 5 + [0] * 4, [1] * 9], device="cuda")
    flash = cross(hidden, enc, enc_mask, None, q_mask)
    sdpa = cross(hidden, enc, enc_mask, None, None)
    m = q_mask.bool()
    assert torch.allclose(flash[m].float(), sdpa[m].float(), atol=3e-2, rtol=3e-2)
    assert float(flash[~m].abs().max()) == 0.0


def _tiny_stitched():
    import torch
    from transformers.models.t5gemma2.configuration_t5gemma2 import (
        T5Gemma2DecoderConfig,
        T5Gemma2TextConfig,
    )
    from transformers.models.t5gemma2.modeling_t5gemma2 import (
        T5Gemma2Decoder,
        T5Gemma2LMHead,
        T5Gemma2TextEncoder,
    )

    from src.train.stitched import Stitched
    torch.manual_seed(0)
    ecfg = T5Gemma2TextConfig(vocab_size=64, hidden_size=48, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                              num_key_value_heads=1, head_dim=16, sliding_window=4, layer_types=["sliding_attention", "full_attention"],
                              query_pre_attn_scalar=16, attn_implementation="sdpa", dropout_rate=0.0, attention_dropout=0.0)
    dcfg = T5Gemma2DecoderConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=2,
                                 num_key_value_heads=1, head_dim=16, sliding_window=4, layer_types=["sliding_attention", "full_attention"],
                                 query_pre_attn_scalar=16, attn_implementation="sdpa", dropout_rate=0.0, attention_dropout=0.0)
    enc, dec = T5Gemma2TextEncoder(ecfg).eval(), T5Gemma2Decoder(dcfg).eval()
    head = T5Gemma2LMHead(32, 64)
    return Stitched(enc, dec, head, "enc", "dec", "sdpa", gradient_checkpointing=True).eval()


def test_stitched_maps_encoder_width_to_decoder_width_and_trains_end_to_end():
    import torch
    m = _tiny_stitched()
    enc_ids = torch.tensor([[2, 5, 6, 7, 8], [2, 9, 10, 0, 0]]); enc_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    dec_ids = torch.tensor([[2, 11, 12], [2, 13, 0]]); dec_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    with torch.no_grad():
        z = m.encode(enc_ids, enc_mask)
        h = m(enc_ids, enc_mask, dec_ids, dec_mask)
    assert z.shape == (2, 5, 32) and h.shape == (2, 3, 32) and m.lm_head(h).shape == (2, 3, 64)
    # padded encoder keys are never attended: changing padded tokens changes nothing
    enc_ids2 = enc_ids.clone(); enc_ids2[1, 3:] = 33
    with torch.no_grad():
        h2 = m(enc_ids2, enc_mask, dec_ids, dec_mask)
    assert torch.allclose(h[dec_mask.bool()], h2[dec_mask.bool()], atol=1e-5)
    m.train()
    out = m(enc_ids, enc_mask, dec_ids, dec_mask)
    m.lm_head(out)[dec_mask.bool()].float().pow(2).mean().backward()
    assert m.stitch.weight.grad.abs().sum() > 0 and m.encoder.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert m.decoder.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0


def test_fit_affine_recovers_a_known_map_and_explained_variance_reads_it():
    import torch

    from src.train.stitched import explained_variance, fit_affine
    g = torch.Generator().manual_seed(0)
    w_true = torch.randn(6, 10, generator=g, dtype=torch.float64); b_true = torch.randn(6, generator=g, dtype=torch.float64)
    x = torch.randn(4000, 10, generator=g, dtype=torch.float64)
    y = x @ w_true.T + b_true + 0.01 * torch.randn(4000, 6, generator=g, dtype=torch.float64)
    w, b = fit_affine(x, y, ridge=1e-6)
    assert w.shape == (6, 10) and b.shape == (6,)
    assert torch.allclose(w, w_true, atol=1e-2) and torch.allclose(b, b_true, atol=1e-2)
    assert explained_variance(x, y, w, b) > 0.999
    assert explained_variance(x, y, torch.zeros_like(w), y.mean(0)) < 1e-6  # the mean explains nothing


def test_sharded_stitched_root_forward_is_the_task_loss():
    import torch

    from src.train.train import ShardedStitched
    m = _tiny_stitched()
    root = ShardedStitched(m)  # unsharded here: the wrapper's forward is what FSDP2 calls as the root
    b = {"enc_ids": torch.tensor([[2, 5, 6, 7, 8]]), "enc_mask": torch.ones(1, 5, dtype=torch.long),
         "dec_ids": torch.tensor([[2, 11, 12]]), "dec_mask": torch.ones(1, 3, dtype=torch.long), "labels": torch.tensor([[11, 12, 1]])}
    loss, n = root(b, torch.device("cpu"))
    assert n == 3 and loss.ndim == 0 and torch.isfinite(loss) and loss > 0
    assert root.config is m.config and root.hf is m


def test_freeze_named_freezes_only_the_named_parameters():
    import torch

    from src.train.train import freeze_named
    m = torch.nn.ModuleDict({"embed_tokens": torch.nn.Linear(4, 4, bias=False), "layer": torch.nn.Linear(4, 4, bias=False)})
    n_on, n_off = freeze_named(m, ["embed_tokens"])
    assert n_on == 16 and n_off == 16
    assert not m["embed_tokens"].weight.requires_grad and m["layer"].weight.requires_grad


def test_stitched_load_retties_a_head_missing_from_an_fsdp_export(tmp_path, monkeypatch):
    import torch

    from src.train import stitched as st
    m = _tiny_stitched()
    m.lm_head.out_proj.weight = m.decoder.embed_tokens.weight  # tied, as in the real decoder
    sd = {k: v for k, v in m.state_dict().items() if k != "lm_head.out_proj.weight"}  # what an FSDP gather produces
    torch.save(sd, tmp_path / "stitched.pt")
    (tmp_path / "stitched.json").write_text('{"enc_base": "enc", "dec_base": "dec", "attn": "sdpa"}')
    fresh = _tiny_stitched()
    monkeypatch.setattr(st.Stitched, "from_pretrained", classmethod(lambda cls, *a, **k: fresh))
    loaded = st.Stitched.load(tmp_path)
    assert loaded.lm_head.out_proj.weight.data_ptr() == loaded.decoder.embed_tokens.weight.data_ptr()
    assert torch.equal(loaded.decoder.embed_tokens.weight, m.decoder.embed_tokens.weight)
    assert torch.equal(loaded.stitch.weight, m.stitch.weight)


def _tiny_soft():
    import torch
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    from transformers.models.t5gemma2.configuration_t5gemma2 import T5Gemma2TextConfig
    from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2TextEncoder

    from src.train.soft import SoftStitched
    torch.manual_seed(0)
    dcfg = Gemma3TextConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=16, sliding_window=4, layer_types=["sliding_attention", "full_attention", "sliding_attention"],
                            query_pre_attn_scalar=16, attn_implementation="sdpa")
    dec = Gemma3ForCausalLM(dcfg).eval()
    ecfg = T5Gemma2TextConfig(vocab_size=64, hidden_size=48, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                              num_key_value_heads=1, head_dim=16, sliding_window=4, layer_types=["sliding_attention", "full_attention"],
                              query_pre_attn_scalar=16, attn_implementation="sdpa", dropout_rate=0.0, attention_dropout=0.0)
    enc = T5Gemma2TextEncoder(ecfg).eval()
    plain = Gemma3ForCausalLM(dcfg).eval()
    plain.load_state_dict(dec.state_dict())
    return SoftStitched(dec, enc, "dec", "enc", "sdpa", gradient_checkpointing=True).eval(), plain


def test_soft_replaces_the_prompt_embeddings_only_and_trains_end_to_end():
    import torch
    m, plain = _tiny_soft()
    enc_ids = torch.tensor([[2, 5, 6, 7, 8], [2, 9, 10, 0, 0]]); enc_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    dec_ids = torch.tensor([[2, 5, 6, 7, 8, 11, 12], [2, 9, 10, 11, 12, 0, 0]]); dec_mask = torch.tensor([[1] * 7, [1] * 5 + [0, 0]])
    with torch.no_grad():
        z = m.encode(enc_ids, enc_mask)
        assert z.shape == (2, 5, 32)  # 48-wide encoder states mapped to the decoder's 32
        # with the decoder's own scaled embeddings in place of the stitched states, the model is the plain decoder exactly:
        # the placement (prompt positions only, real tokens only) and the scaling are what this checks
        h_same = m.decode(m.inputs(m.embed(enc_ids), enc_mask, dec_ids), dec_mask)
        ref = plain.model(input_ids=dec_ids, attention_mask=dec_mask).last_hidden_state
        assert torch.allclose(h_same[dec_mask.bool()], ref[dec_mask.bool()], atol=1e-5)
        h = m(enc_ids, enc_mask, dec_ids, dec_mask)
    assert h.shape == (2, 7, 32) and not torch.allclose(h[dec_mask.bool()], ref[dec_mask.bool()], atol=1e-3)
    # padded encoder positions never reach the decoder: changing them changes nothing
    enc_ids2 = enc_ids.clone(); enc_ids2[1, 3:] = 33
    with torch.no_grad():
        h2 = m(enc_ids2, enc_mask, dec_ids, dec_mask)
    assert torch.allclose(h[dec_mask.bool()], h2[dec_mask.bool()], atol=1e-5)
    m.train()
    out = m(enc_ids, enc_mask, dec_ids, dec_mask)
    assert torch.allclose(out[dec_mask.bool()], h[dec_mask.bool()], atol=1e-5)  # checkpointing changes no value
    m.lm_head(out)[dec_mask.bool()].float().pow(2).mean().backward()
    assert m.stitch.weight.grad.abs().sum() > 0 and m.encoder.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert m.decoder.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert m.decoder.embed_tokens.weight.grad.abs().sum() > 0  # the answer tokens still enter through the embeddings


def test_soft_cached_decoding_matches_the_full_forward():
    import torch
    from transformers import DynamicCache
    m, _ = _tiny_soft()
    enc_ids = torch.tensor([[2, 5, 6, 7, 8], [2, 9, 10, 0, 0]]); enc_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    dec_ids = torch.tensor([[2, 5, 6, 7, 8, 11, 12], [2, 9, 10, 11, 12, 0, 0]]); dec_mask = torch.tensor([[1] * 7, [1] * 5 + [0, 0]])
    with torch.no_grad():
        full = m(enc_ids, enc_mask, dec_ids, dec_mask)
        enc = m.encode(enc_ids, enc_mask)
        # generation's layout: the prompt left-padded, stitched states where the prompt sits, then two answer tokens one at a time
        lens = enc_mask.sum(1).tolist(); n = max(lens)
        ids = torch.full((2, n), 0, dtype=torch.long); mask = torch.zeros((2, n), dtype=torch.long); pos = torch.zeros((2, n), dtype=torch.long)
        emb = m.embed(ids)
        for r, ln in enumerate(lens):
            ids[r, n - ln:] = enc_ids[r, :ln]; mask[r, n - ln:] = 1; pos[r, n - ln:] = torch.arange(ln)
        emb = m.embed(ids)
        for r, ln in enumerate(lens):
            emb[r, n - ln:] = enc[r, :ln]
        cache = DynamicCache(config=m.config)
        h = m.decode(emb, mask, pos, cache)
        steps = [h[:, -1]]
        nxt_pos = pos[:, -1] + 1
        for t in (11, 12):
            mask = torch.cat([mask, torch.ones((2, 1), dtype=torch.long)], 1)
            h = m.decode(m.embed(torch.full((2, 1), t)), mask, nxt_pos[:, None], cache)
            steps.append(h[:, -1]); nxt_pos = nxt_pos + 1
    for r, ln in enumerate(lens):  # prefill's last state = the last prompt position; then the two answer positions
        assert torch.allclose(steps[0][r], full[r, ln - 1], atol=1e-4)
        assert torch.allclose(steps[1][r], full[r, ln], atol=1e-4)
        assert torch.allclose(steps[2][r], full[r, ln + 1], atol=1e-4)


def test_soft_save_and_load_round_trip(tmp_path, monkeypatch):
    import torch

    from src.train import soft as soft_mod
    m, _ = _tiny_soft()
    m.save(tmp_path / "final")
    fresh, _ = _tiny_soft()
    torch.nn.init.normal_(fresh.stitch.weight)
    monkeypatch.setattr(soft_mod.SoftStitched, "from_pretrained", classmethod(lambda cls, *a, **k: fresh))
    loaded = soft_mod.SoftStitched.load(tmp_path / "final")
    assert torch.equal(loaded.stitch.weight, m.stitch.weight) and soft_mod.SoftStitched.is_soft_dir(tmp_path / "final")


def test_soft_stitch_fit_onto_embeddings_starts_the_decoder_near_its_own_input():
    """The E5 stitch is fitted onto the decoder's scaled embeddings of the same tokens: on a synthetic encoder whose
    states are an affine function of those embeddings, the fitted map recovers them and the soft model equals the plain
    decoder."""
    import torch

    from src.train.stitched import fit_affine
    m, plain = _tiny_soft()
    torch.manual_seed(1)
    a = torch.randn(48, 32); c = torch.randn(48)
    ids = torch.randint(3, 64, (400,))
    target = m.embed(ids).detach()
    source = target @ a.T + c  # what a 48-wide "encoder" would emit for these tokens
    w, b = fit_affine(source, target, ridge=1e-6)
    with torch.no_grad():
        m.stitch.weight.copy_(w); m.stitch.bias.copy_(b)
    dec_ids = torch.tensor([[2, 5, 6, 7, 8, 11, 12]]); dec_mask = torch.ones_like(dec_ids)
    enc_ids = dec_ids[:, :5]; enc_mask = torch.ones_like(enc_ids)
    with torch.no_grad():
        states = m.embed(enc_ids) @ a.T + c
        h = m.decode(m.inputs(m.stitch(states), enc_mask, dec_ids), dec_mask)
        ref = plain.model(input_ids=dec_ids, attention_mask=dec_mask).last_hidden_state
    assert torch.allclose(h, ref, atol=1e-3)


def test_collate_continuation_matches_the_seq2seq_cuts_and_supervises_the_continuation_only():
    import torch

    from src.train.data import (
        collate_continuation,
        collate_encdec,
        seq2seq_target_count,
    )
    ex = [Example(id="a", variant="c", input_ids=np.arange(10, 30, dtype=np.int32), n_prompt=0),
          Example(id="b", variant="c", input_ids=np.arange(100, 108, dtype=np.int32), n_prompt=0)]
    b = collate_continuation(ex, [0, 1], pad_id=0, seed=1, max_target=3, cut=(0.5, 0.5))
    # the cut is a property of the document under the seed, not of the batch it lands in
    from src.train.data import seq2seq_cuts
    assert seq2seq_cuts(ex, [0, 1], 7, 3, (0.2, 0.8)) == [seq2seq_cuts(ex, [1, 0], 7, 3, (0.2, 0.8))[1], seq2seq_cuts(ex, [1], 7, 3, (0.2, 0.8))[0]]
    assert seq2seq_cuts(ex, [0], 7, 3, (0.2, 0.8)) != seq2seq_cuts(ex, [0], 8, 3, (0.2, 0.8))
    # document a is cut at 10: prefix 10..19 as tokens, continuation capped at 3 tokens (20, 21, 22), loss there only
    assert b["input_ids"][0].tolist()[:13] == list(range(10, 23)) and b["prompt_len"][0] == 10
    assert b["labels"][0].tolist()[:13] == [-100] * 10 + [20, 21, 22] and b["labels"][0, 13:].eq(-100).all()
    assert int((b["labels"] != -100).sum()) == seq2seq_target_count(ex, [0, 1], 1, 3, (0.5, 0.5))
    # the same cut points as the encoder-decoder objective, so the two arms see the same continuation
    s = collate_encdec(ex, [0, 1], pad_id=0, start_id=999, seed=1, seq2seq=True, max_target=3, cut=(0.5, 0.5))
    assert torch.equal(s["labels"][:, :3], b["labels"][:, 10:13][:, :3]) or s["labels"][0].tolist()[:3] == [20, 21, 22]


def test_collate_encdec_seq2seq_with_the_prefix_on_the_decoder_side():
    from src.train.data import collate_encdec
    ex = [Example(id="a", variant="c", input_ids=np.arange(10, 30, dtype=np.int32), n_prompt=0)]
    # E5's adaptation layout: encoder reads the prefix, the decoder side is the prefix (its positions carry the
    # stitched states) then the continuation, no start token, labels on the continuation only
    s = collate_encdec(ex, [0], pad_id=0, start_id=999, seed=1, seq2seq=True, max_target=3, cut=(0.5, 0.5), dec_prompt=True, dec_start=False)
    assert s["enc_ids"][0].tolist() == list(range(10, 20)) and s["enc_mask"][0].sum() == 10
    assert s["dec_ids"][0].tolist() == list(range(10, 20)) + [20, 21] and s["labels"][0].tolist() == [-100] * 9 + [20, 21, 22]
    # without dec_prompt (the Qwen E3 adaptation) the decoder side is the start token and the continuation, as before
    q = collate_encdec(ex, [0], pad_id=0, start_id=999, seed=1, seq2seq=True, max_target=3, cut=(0.5, 0.5))
    assert q["dec_ids"][0].tolist() == [999, 20, 21] and q["labels"][0].tolist() == [20, 21, 22]


def test_sharded_root_forward_serves_the_soft_model_too():
    import torch

    from src.train.train import ShardedStitched
    m, _ = _tiny_soft()
    root = ShardedStitched(m)
    batch = {"enc_ids": torch.tensor([[2, 5, 6, 7, 8]]), "enc_mask": torch.ones(1, 5, dtype=torch.long),
             "dec_ids": torch.tensor([[2, 5, 6, 7, 8, 11, 12]]), "dec_mask": torch.ones(1, 7, dtype=torch.long),
             "labels": torch.tensor([[-100, -100, -100, -100, 11, 12, 13]])}
    loss, n = root(batch, torch.device("cpu"))
    assert n == 3 and loss.ndim == 0 and torch.isfinite(loss)


def test_soft_load_reties_a_head_missing_from_an_fsdp_export(tmp_path, monkeypatch):
    import torch

    from src.train import soft as soft_mod
    m, _ = _tiny_soft()
    sd = {k: v for k, v in m.state_dict().items() if k != "lm_head.weight"}  # what an FSDP gather of tied weights carries
    (tmp_path / "final").mkdir()
    torch.save(sd, tmp_path / "final" / "soft.pt")
    (tmp_path / "final" / "soft.json").write_text('{"dec_base": "dec", "enc_base": "enc", "attn": "sdpa"}')
    fresh, _ = _tiny_soft()
    monkeypatch.setattr(soft_mod.SoftStitched, "from_pretrained", classmethod(lambda cls, *a, **k: fresh))
    loaded = soft_mod.SoftStitched.load(tmp_path / "final")
    assert loaded.lm_head.weight.data_ptr() == loaded.decoder.embed_tokens.weight.data_ptr()
    assert torch.equal(loaded.decoder.embed_tokens.weight, m.decoder.embed_tokens.weight)


def test_plan_counts_no_prediction_for_the_first_token_of_a_promptless_document():
    ex = [Example(id="a", variant="c", input_ids=np.arange(10, 30, dtype=np.int32), n_prompt=0),
          Example(id="b", variant="c", input_ids=np.arange(100, 108, dtype=np.int32), n_prompt=3)]
    steps = plan_epoch(ex, 2, 8192, 1, 0, 0)
    assert len(steps) == 1 and steps[0].target_tokens == 19 + 5  # the shifted labels of a and the answer of b


def _ddp_worker(rank: int, world: int, port: int, q):
    import os

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    from src.train.train import LossWrapper, target_loss, unwrap
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    cfg = Gemma3TextConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=2,
                           num_key_value_heads=1, head_dim=16, sliding_window=4, layer_types=["sliding_attention", "full_attention"],
                           query_pre_attn_scalar=16, attn_implementation="sdpa")
    model = Gemma3ForCausalLM(cfg)  # same seed, same weights on both ranks
    model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    wrapped = DDP(LossWrapper(model))
    assert unwrap(wrapped) is model and set(wrapped.module.state_dict()) == set(model.state_dict())  # no `inner.` prefix
    ids = torch.arange(1 + 20 * rank, 9 + 20 * rank).view(1, -1)  # a different micro-batch per rank
    labels = ids.clone()
    labels[:, :3] = -100
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels, "prompt_len": torch.tensor([3])}
    loss, n = target_loss(wrapped, batch, torch.device("cpu"))
    assert n == 5
    (loss / n).backward()
    g = model.model.layers[0].mlp.down_proj.weight.grad.clone()
    # the gradient each rank would have on its own: the same forward on an unwrapped copy
    dist.barrier()
    alone = Gemma3ForCausalLM(cfg)
    alone.load_state_dict(model.state_dict())
    l2, _ = target_loss(alone, batch, torch.device("cpu"))
    (l2 / n).backward()
    own = alone.model.layers[0].mlp.down_proj.weight.grad
    gathered = [torch.zeros_like(own) for _ in range(world)]
    dist.all_gather(gathered, own)
    q.put((rank, torch.allclose(g, sum(gathered) / world, atol=1e-5), torch.allclose(g, own, atol=1e-5)))
    dist.destroy_process_group()


def test_ddp_routes_the_loss_through_the_reducer_so_gradients_are_averaged_across_ranks():
    """The 2026-09-28 bug: the forward on the bare module never armed DDP's reducer, so ranks trained apart."""
    import socket

    import torch.multiprocessing as mp
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_ddp_worker, args=(r, 2, port, q)) for r in range(2)]
    for p in procs:
        p.start()
    results = [q.get(timeout=300) for _ in procs]
    for p in procs:
        p.join(60)
    assert all(p.exitcode == 0 for p in procs)
    assert all(averaged for _, averaged, _ in results), results
    assert not any(own for _, _, own in results), results  # the ranks' own gradients differ, so averaging is observable


def test_adaptation_documents_open_with_the_tokenizer_start_token():
    from src.train.data import _encode_docs, start_tokens

    class Tok:  # Gemma-like: a start token in front of every prompt
        bos_token_id = 2

        def __call__(self, texts, add_special_tokens=True):
            enc = lambda t: [ord(c) for c in t]
            return {"input_ids": ([2] + enc(texts) if add_special_tokens else enc(texts)) if isinstance(texts, str)
                    else [([2] if add_special_tokens else []) + enc(t) for t in texts]}

    class NoBos(Tok):
        bos_token_id = None

    assert start_tokens(Tok()) == [2] and start_tokens(NoBos()) == []
    out: list = []
    assert _encode_docs([("d", "abcdef")], Tok(), max_seq_len=6, eos=1, out=out) == 1  # truncated to fit the start token and eos
    assert out[0].input_ids.tolist() == [2, 97, 98, 99, 100, 1] and out[0].n_prompt == 0
    out = []
    _encode_docs([("d", "ab")], NoBos(), max_seq_len=6, eos=1, out=out)
    assert out[0].input_ids.tolist() == [97, 98, 1]
