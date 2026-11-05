"""E5-merged-attention (src/train/merged.py): a Gemma 3 decoder reading a T5Gemma 2 encoder through merged attention."""

import pytest

torch = pytest.importorskip("torch")

LAYER_TYPES = ["sliding_attention", "full_attention", "sliding_attention"]


def _tiny(stitch=None, seed: int = 0, d_enc: int = 48):
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    from transformers.models.t5gemma2.configuration_t5gemma2 import T5Gemma2TextConfig
    from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2TextEncoder

    from src.train.merged import MergedAttention
    torch.manual_seed(0)
    dcfg = Gemma3TextConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=16, sliding_window=3, layer_types=LAYER_TYPES,
                            query_pre_attn_scalar=16, attn_implementation="sdpa")
    dec = Gemma3ForCausalLM(dcfg).eval()
    ecfg = T5Gemma2TextConfig(vocab_size=64, hidden_size=d_enc, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                              num_key_value_heads=1, head_dim=16, sliding_window=4, layer_types=["sliding_attention", "full_attention"],
                              query_pre_attn_scalar=16, attn_implementation="sdpa", dropout_rate=0.0, attention_dropout=0.0)
    enc = T5Gemma2TextEncoder(ecfg).eval()
    return MergedAttention(dec, enc, "dec", "enc", "sdpa", gradient_checkpointing=False, stitch=stitch, seed=seed).eval()


ENC = (torch.tensor([[2, 5, 6, 7, 8, 9], [2, 9, 10, 11, 0, 0]]), torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]))
DEC = (torch.tensor([[2, 11, 12, 13, 14], [2, 13, 15, 0, 0]]), torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]))


def test_matches_t5gemma2_merged_attention_with_the_same_weights():
    """The Gemma 3 decoder under merged attention computes what T5Gemma 2's own decoder computes with those weights and
    the stitched states as its encoder output: the implementation is the reference's, not an approximation of it."""
    from transformers.models.t5gemma2.configuration_t5gemma2 import (
        T5Gemma2DecoderConfig,
    )
    from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2Decoder
    m = _tiny()
    g = m.decoder.config
    rcfg = T5Gemma2DecoderConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=2,
                                 num_key_value_heads=1, head_dim=16, sliding_window=3, layer_types=LAYER_TYPES,
                                 query_pre_attn_scalar=16, attn_implementation="sdpa", dropout_rate=0.0, attention_dropout=0.0,
                                 rope_parameters=g.rope_parameters, rms_norm_eps=g.rms_norm_eps,
                                 attn_logit_softcapping=g.attn_logit_softcapping, hidden_activation=g.hidden_activation)
    ref = T5Gemma2Decoder(rcfg).eval()
    rename = {"input_layernorm": "pre_self_attn_layernorm", "post_attention_layernorm": "post_self_attn_layernorm"}
    sd = {}
    for k, v in m.decoder.state_dict().items():
        for a, b in rename.items():
            k = k.replace(a, b)
        sd[k] = v
    missing, unexpected = ref.load_state_dict(sd, strict=False)
    assert not unexpected and all("eoi" in k for k in missing), (missing, unexpected)
    (ei, em), (di, dm) = ENC, DEC
    with torch.no_grad():
        ours = m(ei, em, di, dm)
        states = m.encode(ei, em)
        theirs = ref(input_ids=di, attention_mask=dm, encoder_hidden_states=states, encoder_attention_mask=em).last_hidden_state
    real = dm.bool()
    torch.testing.assert_close(ours[real], theirs[real], rtol=1e-4, atol=1e-4)


def test_causal_padding_and_encoder_dependence():
    m = _tiny()
    (ei, em), (di, dm) = ENC, DEC
    with torch.no_grad():
        base = m(ei, em, di, dm)
        later = di.clone(); later[0, 3] = 20  # a later decoder token cannot change earlier positions
        torch.testing.assert_close(m(ei, em, later, dm)[0, :3], base[0, :3])
        padded = ei.clone(); padded[1, 4:] = 33  # what sits under encoder padding is never read
        torch.testing.assert_close(m(padded, em, di, dm)[1, :3], base[1, :3])
        other = ei.clone(); other[0, 2] = 40  # every real encoder position is read, by every decoder position
        assert not torch.allclose(m(other, em, di, dm)[0, 0], base[0, 0])


def test_incremental_decoding_with_a_cache_matches_the_full_pass():
    from transformers import DynamicCache
    m = _tiny()
    (ei, em), (di, _) = ENC, DEC
    ei, em, di = ei[:1], em[:1], di[:1]
    with torch.no_grad():
        full = m(ei, em, di, torch.ones_like(di))
        m.set_encoder(m.encode(ei, em))
        with pytest.raises(ValueError):  # a config-built cache trims the sliding layers; refused
            m.decode(di[:, :1], torch.ones((1, 1), dtype=torch.long), em, torch.tensor([[0]]), DynamicCache(config=m.config))
        cache = DynamicCache()
        steps = []
        for t in range(di.shape[1]):
            pos = torch.tensor([[t]])
            steps.append(m.decode(di[:, t:t + 1], torch.ones((1, t + 1), dtype=torch.long), em, pos, cache)[:, -1])
    torch.testing.assert_close(torch.stack(steps, 1), full, rtol=1e-4, atol=1e-4)


def test_learned_start_is_seeded_and_fitted_start_loads_one_map(tmp_path):
    a, b, c = _tiny(seed=0), _tiny(seed=0), _tiny(seed=1)
    assert torch.equal(a.stitch.weight, b.stitch.weight) and not torch.equal(a.stitch.weight, c.stitch.weight)
    from src.train.merged import load_stitch
    w, bias = torch.randn(32, 48), torch.randn(32)
    torch.save({"weight": w, "bias": bias}, tmp_path / "fit.pt")
    fitted = _tiny(stitch=load_stitch(tmp_path / "fit.pt"))
    assert torch.allclose(fitted.stitch.weight.float(), w, atol=2e-2)
    torch.save({"weights": w[None], "biases": bias[None]}, tmp_path / "layers.pt")
    with pytest.raises(ValueError):  # not a single map
        load_stitch(tmp_path / "layers.pt")
    with pytest.raises(ValueError):  # the wrong widths for the pair
        _tiny(stitch=torch.nn.Linear(40, 32))


def test_gradients_reach_encoder_stitch_and_decoder_and_checkpointing_changes_nothing():
    (ei, em), (di, dm) = ENC, DEC
    grads = []
    for gc in (False, True):
        m = _tiny().train()
        if gc:
            for tower in (m.encoder, m.decoder):
                tower.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        h = m(ei, em, di, dm)
        m.lm_head(h)[dm.bool()].logsumexp(-1).sum().backward()
        g = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
        assert any(n.startswith("encoder.") for n in g) and "stitch.weight" in g
        assert any(".self_attn.k_proj" in n for n in g)
        grads.append(g)
    for n in grads[0]:
        torch.testing.assert_close(grads[0][n], grads[1][n], rtol=1e-3, atol=1e-5)


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    from src.train import merged
    m = _tiny()
    m.save(tmp_path)
    monkeypatch.setattr(merged.MergedAttention, "from_pretrained",
                        classmethod(lambda cls, *a, **k: _tiny(seed=7)))  # a different start, overwritten by the load
    back = merged.MergedAttention.load(tmp_path, gradient_checkpointing=False)
    (ei, em), (di, dm) = ENC, DEC
    with torch.no_grad():
        torch.testing.assert_close(back(ei, em, di, dm), m(ei, em, di, dm))
    assert merged.MergedAttention.is_merged_dir(tmp_path)


def test_flash_attention_is_refused():
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    from src.train.merged import MergedAttention
    dec = Gemma3ForCausalLM(Gemma3TextConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=1,
                                             num_attention_heads=2, num_key_value_heads=1, head_dim=16))
    for attn in ("flash_attention_2", "eager"):  # eager would add the boolean mask to the scores and mask nothing
        with pytest.raises(ValueError):
            MergedAttention(dec, None, "d", "e", attn)


def _merged_cfg():
    import yaml

    from src.train.config import TrainConfig
    path = "configs/adapt/continuation-corpora-e5-merged-attention-learned-t5gemma2-270m-enc-gemma3-270m.yaml"
    return TrainConfig.model_validate(yaml.safe_load(open(path).read()))


def test_trainer_builds_the_merged_arch_and_its_loss_trains_the_stitch(monkeypatch):
    """load_model's merged branch (from the learned pilot config) and model_loss on both batch layouts the arch meets:
    the adaptation stage's continuation and the fine-tune's prompt / answer, each with a <bos>-started answer-only decoder."""
    import numpy as np
    import transformers

    from src.train import merged
    from src.train import train as T
    from src.train.data import Example, collate_encdec
    seen = {}

    def fake(cls, dec, enc, attn, gc, stitch, seed=0):
        seen.update(dec=dec, enc=enc, attn=attn, stitch=stitch, seed=seed)
        return _tiny(seed=seed)

    class Tok:
        bos_token_id, eos_token_id, pad_token_id = 2, 1, 0

    monkeypatch.setattr(merged.MergedAttention, "from_pretrained", classmethod(fake))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: Tok()))
    cfg = _merged_cfg()
    _, m, attn = T.load_model(cfg, torch.device("cpu"))
    assert isinstance(m, merged.MergedAttention) and attn == "sdpa"
    assert seen == {"dec": "google/gemma-3-270m", "enc": "google/t5gemma-2-270m-270m", "attn": "sdpa", "stitch": None, "seed": 0}
    m.train()
    cont = [Example(id="a", variant="c", input_ids=np.arange(3, 40, dtype=np.int32) % 60 + 3, n_prompt=0),
            Example(id="b", variant="c", input_ids=np.arange(5, 30, dtype=np.int32) % 60 + 3, n_prompt=0)]
    task = [Example(id="a", variant="c", input_ids=np.arange(3, 20, dtype=np.int32), n_prompt=12),
            Example(id="b", variant="c", input_ids=np.arange(5, 15, dtype=np.int32), n_prompt=6)]
    for ex, seq2seq in ((cont, True), (task, False)):
        b = collate_encdec(ex, [0, 1], pad_id=0, start_id=2, seed=1, seq2seq=seq2seq, max_target=8, cut=(0.5, 0.5))
        assert (b["dec_ids"][:, 0] == 2).all()  # the answer-only sequence opens with <bos>
        loss, n = T.model_loss(m, b, torch.device("cpu"))
        assert torch.isfinite(loss) and n > 0
        loss.backward()
        assert m.stitch.weight.grad.abs().sum() > 0
        m.zero_grad()


def test_greedy_cached_decoding_equals_the_argmax_of_a_full_pass():
    """generate_merged's loop on CPU: greedy steps with an untrimmed cache give the tokens a full forward over the
    generated prefix predicts, past the sliding window."""
    from transformers import DynamicCache
    m = _tiny()
    ei, em = ENC
    with torch.no_grad():
        m.set_encoder(m.encode(ei, em))
        cache = DynamicCache()
        nxt = torch.full((2,), 2)
        pos = torch.zeros((2, 1), dtype=torch.long)
        dm = torch.ones((2, 1), dtype=torch.long)
        seq = [nxt]
        for _ in range(6):
            h = m.decode(nxt[:, None], dm, em, pos, cache)
            nxt = m.lm_head(h[:, -1]).argmax(-1)
            seq.append(nxt)
            dm = torch.cat([dm, torch.ones((2, 1), dtype=torch.long)], 1)
            pos = pos + 1
        seq = torch.stack(seq, 1)
        full = m.lm_head(m(ei, em, seq[:, :-1], torch.ones_like(seq[:, :-1]))).argmax(-1)
    assert torch.equal(full, seq[:, 1:])


def test_one_map_onto_the_layer_average_is_the_least_squares_map_for_every_layer_at_once():
    """Fitting onto the mean of the layer targets gives the same map as fitting one map to all the layers' targets stacked
    (the normal equations coincide), and the streamed per-layer variance matches the direct computation."""
    from src.train.fit_stitch import StreamedVariance
    from src.train.stitched import AffineMoments, explained_variance, fit_affine
    torch.manual_seed(0)
    x = torch.randn(3000, 6, dtype=torch.float64)
    ys = [x @ torch.randn(3, 6, dtype=torch.float64).T + torch.randn(3, dtype=torch.float64) for _ in range(3)]
    m = AffineMoments(6, 3)
    for i in range(0, 3000, 1000):
        m.add(x[i:i + 1000], torch.stack([y[i:i + 1000] for y in ys]).mean(0))
    w, b = m.solve(ridge=1e-9)
    w_all, b_all = fit_affine(torch.cat([x] * 3), torch.cat(ys), ridge=1e-9)
    torch.testing.assert_close(w, w_all, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(b, b_all, atol=1e-6, rtol=1e-6)
    sv = StreamedVariance(3)
    for i in range(0, 3000, 700):
        sv.add(x[i:i + 700], [y[i:i + 700] for y in ys], [w] * 3, [b] * 3)
    for ev, y in zip(sv.explained(), ys):
        assert abs(ev - explained_variance(x, y, w, b)) < 1e-9


def test_layer_targets_capture_each_layers_normalised_input(tmp_path):
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    from src.train.fit_stitch import LayerTargets
    torch.manual_seed(0)
    cfg = Gemma3TextConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=2,
                           num_key_value_heads=1, head_dim=16, sliding_window=3, layer_types=LAYER_TYPES,
                           query_pre_attn_scalar=16)
    Gemma3ForCausalLM(cfg).save_pretrained(tmp_path)
    t = LayerTargets(str(tmp_path), "sdpa").float().eval()
    ids, mask = torch.tensor([[2, 5, 6, 7], [2, 8, 0, 0]]), torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    out = t.layer_states(ids, mask)
    assert len(out) == 3 and all(o.shape == (6, 32) for o in out)  # real tokens only
    # layer 0's normalised input is the input norm of the (scaled) embeddings
    want = t.text.layers[0].input_layernorm(t.text.embed_tokens(ids))[mask.bool()].float()
    torch.testing.assert_close(out[0], want, rtol=1e-4, atol=1e-4)
    assert not torch.allclose(out[1], out[0])  # each layer its own target
