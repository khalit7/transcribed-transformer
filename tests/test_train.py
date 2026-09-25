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
