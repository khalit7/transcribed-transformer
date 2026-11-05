"""E5-merged-attention: a T5Gemma 2 encoder read by a decoder-only Gemma 3 through merged attention.

The decoder is a Gemma 3 causal language model, its weights as pretrained. Its decoder sequence holds only the answer
(a start token, then the target), as a native T5Gemma 2 decoder's does, and every decoder layer reads the encoder the
way T5Gemma 2's merged attention does (transformers' `T5Gemma2MergedAttention`): the layer's own key and value
projections, and its key norm, are applied to encoder states as well as to the decoder's own positions; the encoder
keys get no rotary position (order lives inside the encoder states); keys and values are concatenated, and one
attention runs over both, causal (and sliding-window in the sliding layers) over the decoder's positions and open over
every real encoder position. No attention weights are added.

What is added is the stitch: one affine map, applied once to the encoder's final states, from the encoder's width to the
decoder's; every decoder layer reads the same stitched states through its own projections, as every T5Gemma 2 decoder
layer reads the same encoder output (and as E4-mix-and-match's single stitch feeds its decoder). Two starts, chosen in
the config:

- `model.stitch: null` -- the map starts fresh (seeded, `nn.Linear`'s default initialisation) and is learned entirely in
  the adaptation stage and the fine-tune ("learned");
- `model.stitch: <path>` -- the map starts from a ridge regression (`src/train/fit_stitch.py --target-layer-mean`): from
  the encoder's state of each prompt token onto the average, over Gemma 3's layers, of the normalised hidden state (the
  output of each layer's input norm, what its key and value projections read) that Gemma 3 computes when it reads the
  prompt as tokens. Least squares onto that average is the single map closest to every layer's input at once ("fitted");
  it then trains with everything else.

Gemma 3 runs under SDPA here and nowhere else: the merged mask is boolean, which SDPA reads as allowed / masked;
FlashAttention takes no such mask, and the eager path would add it to the scores.

Checkpoint layout (`MergedAttention.save` / `.load`): `merged.pt` (state dict), `merged.json` (decoder base, encoder
base, attention implementation), tokenizer files.
"""

import json
import types
from pathlib import Path

import torch
from torch import nn


def _merged_attention_forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None,
                              **kwargs):
    """A Gemma3Attention.forward that also attends to `self._merged_enc` (this layer's stitched encoder states [B, S, d])
    through the layer's own projections, as T5Gemma2MergedAttention does. `attention_mask` is the merged boolean mask
    [B, 1, T, K + S] built by `MergedAttention.decode` (K: the decoder keys, cached ones included). The encoder keys and
    values are computed here, inside the layer, so a sharded layer (FSDP) projects them with its own gathered weights;
    during generation (a cache given) they are computed once per batch and kept."""
    from transformers.models.gemma3.modeling_gemma3 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2))
    k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2))
    v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    if past_key_values is not None:
        k, v = past_key_values.update(k, v, self.layer_idx)
    kv = self._merged_kv if past_key_values is not None else None
    if kv is None:
        enc = self._merged_enc
        enc_shape = (*enc.shape[:-1], -1, self.head_dim)
        kv = (self.k_norm(self.k_proj(enc).view(enc_shape).transpose(1, 2)),  # no rotary on encoder keys, as T5Gemma 2
              self.v_proj(enc).view(enc_shape).transpose(1, 2))
        if past_key_values is not None:
            self._merged_kv = kv
    ck, cv = kv
    k = torch.cat([k, ck], dim=2)
    v = torch.cat([v, cv], dim=2)
    interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
    out, weights = interface(self, q, k, v, attention_mask, dropout=self.attention_dropout if self.training else 0.0,
                             scaling=self.scaling, sliding_window=None, **kwargs)  # the window is in the mask
    out = self.o_proj(out.reshape(*input_shape, -1).contiguous())
    return out, weights


class MergedAttention(nn.Module):
    def __init__(self, dec_lm, text_encoder, dec_base: str, enc_base: str, attn: str = "sdpa",
                 gradient_checkpointing: bool = True, stitch: nn.Linear | None = None, seed: int = 0):
        """dec_lm: a Gemma 3 causal LM; text_encoder: a T5Gemma 2 text encoder (final states after its norm). Built
        modules, so a tiny pair can be tested on CPU; prefer `MergedAttention.from_pretrained`."""
        super().__init__()
        if attn != "sdpa":  # the merged mask is boolean: SDPA reads it as allowed / masked; eager would add it to the
            # scores (0 or 1) and mask nothing, and FlashAttention takes no such mask
            raise ValueError(f"merged attention runs under sdpa only, not {attn}")
        self.dec_base, self.enc_base, self.attn = dec_base, enc_base, attn
        text = dec_lm.model.language_model if hasattr(dec_lm.model, "language_model") else dec_lm.model
        self.config = text.config
        self.config.use_cache = False
        self.config._attn_implementation = attn
        self.decoder = text
        self.lm_head = dec_lm.lm_head
        self.encoder = text_encoder
        n, d_enc, d_dec = text.config.num_hidden_layers, text_encoder.config.hidden_size, text.config.hidden_size
        dt = next(self.decoder.parameters()).dtype
        if stitch is None:  # "learned": a fresh, seeded start
            gen_state = torch.random.get_rng_state()
            torch.manual_seed(seed)
            stitch = nn.Linear(d_enc, d_dec, bias=True, dtype=dt)
            torch.random.set_rng_state(gen_state)
        if (stitch.in_features, stitch.out_features) != (d_enc, d_dec):
            raise ValueError(f"stitch {stitch.in_features} -> {stitch.out_features}, the pair needs {d_enc} -> {d_dec}")
        self.stitch = stitch
        for layer in self.decoder.layers[:n]:
            layer.self_attn.forward = types.MethodType(_merged_attention_forward, layer.self_attn)
            layer.self_attn._merged_enc = None
            layer.self_attn._merged_kv = None
        if gradient_checkpointing:
            for tower in (self.encoder, self.decoder):
                tower.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    @classmethod
    def from_pretrained(cls, dec_base: str, enc_base: str, attn: str = "sdpa", gradient_checkpointing: bool = True,
                        stitch_path: str | Path | None = None, seed: int = 0) -> "MergedAttention":
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
        dec_lm = AutoModelForCausalLM.from_pretrained(dec_base, dtype=torch.bfloat16, attn_implementation=attn)
        full = AutoModelForSeq2SeqLM.from_pretrained(enc_base, dtype=torch.bfloat16, attn_implementation="sdpa")
        enc = full.model.encoder.text_model  # the rest (decoder, image tower, projector) is dropped
        del full
        return cls(dec_lm, enc, dec_base, enc_base, attn, gradient_checkpointing, load_stitch(stitch_path), seed)

    # ---- pieces
    def encode(self, enc_ids: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
        """Stitched encoder states [B, S, d_dec]: what every decoder layer reads."""
        enc = self.encoder(input_ids=enc_ids, attention_mask=enc_mask).last_hidden_state
        return self.stitch(enc.to(self.stitch.weight.dtype))

    def set_encoder(self, enc: torch.Tensor) -> None:
        """Hand every layer the stitched encoder states (once per batch); each layer projects them into keys and values
        itself, with its own weights."""
        for layer in self.decoder.layers[: self.config.num_hidden_layers]:
            layer.self_attn._merged_enc = enc
            layer.self_attn._merged_kv = None  # a new batch: any generation cache of encoder keys is stale

    def _masks(self, dec_mask: torch.Tensor | None, enc_mask: torch.Tensor, position_ids: torch.Tensor, n_past: int,
               t: int) -> dict[str, torch.Tensor]:
        """Merged boolean masks [B, 1, T, K + S] per layer type: causal (and windowed in sliding layers) over the K =
        n_past + T decoder keys, open over every real encoder position."""
        dev = enc_mask.device
        b = enc_mask.shape[0]
        k = n_past + t
        q_pos = position_ids[:, :, None]  # [B, T, 1], absolute decoder positions
        k_pos = torch.arange(k, device=dev)[None, None, :]  # [1, 1, K]
        causal = k_pos <= q_pos
        if dec_mask is not None:  # the decoder keys a row may see (right padding in training; growing ones in generation)
            causal = causal & dec_mask[:, None, :k].bool()
        window = self.config.sliding_window
        sliding = causal & (q_pos - k_pos < window) if window else causal
        cross = enc_mask[:, None, :].bool().expand(b, t, enc_mask.shape[1])
        return {"full_attention": torch.cat([causal, cross], -1)[:, None],
                "sliding_attention": torch.cat([sliding, cross], -1)[:, None]}

    def decode(self, dec_ids: torch.Tensor, dec_mask: torch.Tensor | None, enc_mask: torch.Tensor,
               position_ids: torch.Tensor | None = None, past=None) -> torch.Tensor:
        """Decoder hidden states after the final norm, the answer positions only. `set_encoder` must have run for this
        batch. With `past` (an untrimmed `DynamicCache()`), one or more decoding steps."""
        if past is not None and any(type(layer).__name__ != "DynamicLayer" for layer in past.layers):
            # a cache built from the config trims the sliding layers to their window; the merged mask counts every
            # decoder key and applies the window itself, so the cache must keep them all: DynamicCache() without config
            raise ValueError("merged attention needs an untrimmed cache: DynamicCache() without a config")
        h = self.decoder.embed_tokens(dec_ids)  # Gemma scales inside the module
        n_past = past.get_seq_length() if past is not None else 0
        if position_ids is None:
            position_ids = (torch.arange(dec_ids.shape[1], device=dec_ids.device) + n_past)[None].expand(dec_ids.shape[0], -1)
        masks = self._masks(dec_mask, enc_mask, position_ids, n_past, dec_ids.shape[1])
        pe = {lt: self.decoder.rotary_emb(h, position_ids, lt) for lt in set(self.config.layer_types)}
        for i, layer in enumerate(self.decoder.layers[: self.config.num_hidden_layers]):
            lt = self.config.layer_types[i]
            h = layer(h, attention_mask=masks[lt], position_embeddings=pe[lt], position_ids=position_ids, past_key_values=past)
        return self.decoder.norm(h)

    def forward(self, enc_ids, enc_mask, dec_ids, dec_mask) -> torch.Tensor:
        """Decoder hidden states [B, T, d_dec]; the caller applies lm_head where labels exist."""
        self.set_encoder(self.encode(enc_ids, enc_mask))
        return self.decode(dec_ids, dec_mask, enc_mask)

    # ---- persistence
    def save(self, out: Path, tokenizer=None) -> None:
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out / "merged.pt")
        (out / "merged.json").write_text(json.dumps({"dec_base": self.dec_base, "enc_base": self.enc_base, "attn": self.attn}))
        if tokenizer is not None:
            tokenizer.save_pretrained(out)

    @classmethod
    def load(cls, path: str | Path, attn: str | None = None, gradient_checkpointing: bool = True) -> "MergedAttention":
        path = Path(path)
        meta = json.loads((path / "merged.json").read_text())
        m = cls.from_pretrained(meta["dec_base"], meta["enc_base"], attn or meta["attn"], gradient_checkpointing)
        sd = torch.load(path / "merged.pt", map_location="cpu")
        missing, unexpected = m.load_state_dict(sd, strict=False)
        tied = {"lm_head.weight"}  # tied to the decoder's embeddings; an FSDP-gathered export may carry it once
        if unexpected or set(missing) - tied:
            raise RuntimeError(f"merged checkpoint mismatch: missing {sorted(set(missing) - tied)}, unexpected {sorted(unexpected)}")
        if missing:
            m.lm_head.weight = m.decoder.embed_tokens.weight
        return m

    @staticmethod
    def is_merged_dir(path: str | Path) -> bool:
        return (Path(path) / "merged.json").exists()


def load_stitch(path: str | Path | None) -> nn.Linear | None:
    """The fitted map from `fit_stitch.py` ({"weight": [d_dec, d_enc], "bias": [d_dec]}), or None for a fresh start."""
    if path is None:
        return None
    fit = torch.load(Path(path), map_location="cpu")
    if "weight" not in fit:
        raise ValueError(f"{path}: not a single-map stitch")
    lin = nn.Linear(fit["weight"].shape[1], fit["weight"].shape[0], bias=True, dtype=torch.bfloat16)
    with torch.no_grad():
        lin.weight.copy_(fit["weight"])
        lin.bias.copy_(fit["bias"])
    return lin
