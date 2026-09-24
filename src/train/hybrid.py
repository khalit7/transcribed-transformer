"""E4b: a decoder-only model given an encoder through new cross-attention.

The decoder is a causal language model (Gemma 3 1B) fed the prompt and the answer exactly as in E1: its
self-attention sees every token. The encoder is the bidirectional text encoder of a native encoder-decoder
built from the same family (T5Gemma 2 1b-1b), run over the prompt. Every decoder layer gets a cross-
attention sublayer between self-attention and the MLP (T5 ordering), in Gemma's pre- and post-norm
arrangement: queries, keys, values, the per-head q/k norms and the output projection are copied from that
layer's self-attention, and the sublayer's post-norm weight starts at -1, so its output (1 + w) · rms(x) is
exactly zero: at initialisation the model is the E1 decoder exactly and any change is the encoder's doing.
The gate is the norm weight, not the output projection: under Gemma's post-norm a zero output projection is
zero only at step 0, because the norm rescales whatever the projection produces to unit size after the first
update (found in the smoke: loss 1.6 → 8.4 in three steps, gradient norm 9,000 from the norm's slope at zero).
No rotary embedding on cross-attention. The decoder's own tokenizer serves
both towers (the two checkpoints tokenise identically; checked on real prompts).

The decoder's Hugging Face forward is reused unchanged (masks, rotary, sliding-window layers, cache); its
layers are wrapped, and the wrapper inserts the cross-attention. The encoder states reach the layers as a
context set on each wrapper before the decoder forward. In evaluation the encoder-side keys and values can
be computed once per prompt and reused across decoding steps.

Checkpoint layout (`Hybrid.save` / `Hybrid.load`): `hybrid.pt` (state dict), `hybrid.json` (decoder base,
encoder base and attention implementation), tokenizer files.
"""

import copy
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class CrossAttention(nn.Module):
    """Decoder queries over encoder keys and values; GQA as in the self-attention it is copied from."""

    def __init__(self, self_attn, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv = config.num_key_value_heads
        self.head_dim = self_attn.head_dim
        self.scaling = self_attn.scaling
        self.q_proj = copy.deepcopy(self_attn.q_proj)
        self.k_proj = copy.deepcopy(self_attn.k_proj)
        self.v_proj = copy.deepcopy(self_attn.v_proj)
        self.q_norm = copy.deepcopy(self_attn.q_norm)
        self.k_norm = copy.deepcopy(self_attn.k_norm)
        self.o_proj = copy.deepcopy(self_attn.o_proj)  # the gate is the post-norm weight (HybridLayer), not this

    def kv(self, enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, tk, _ = enc.shape
        k = self.k_norm(self.k_proj(enc).view(b, tk, self.num_kv, self.head_dim)).transpose(1, 2)
        v = self.v_proj(enc).view(b, tk, self.num_kv, self.head_dim).transpose(1, 2)
        return k, v

    def forward(self, hidden: torch.Tensor, enc: torch.Tensor, enc_mask: torch.Tensor,
                kv: tuple[torch.Tensor, torch.Tensor] | None = None, q_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, tq, _ = hidden.shape
        q = self.q_norm(self.q_proj(hidden).view(b, tq, self.num_heads, self.head_dim))  # [B, tq, H, d]
        if kv is None and q_mask is not None and flash_varlen_available(hidden):
            # training at long context: FlashAttention's variable-length kernel over the real tokens of both sides,
            # O(tokens) memory; SDPA with a key mask materialises the [tq, tk] matrix per layer (1.8 GiB at 16k)
            return self.o_proj(self._flash(q, enc, enc_mask, q_mask).reshape(b, tq, self.num_heads * self.head_dim))
        k, v = kv if kv is not None else self.kv(enc)
        mask = enc_mask.bool()[:, None, None, :]  # keys at padding are never attended
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k, v, attn_mask=mask, scale=self.scaling, enable_gqa=True)
        return self.o_proj(out.transpose(1, 2).reshape(b, tq, self.num_heads * self.head_dim))

    def _flash(self, q: torch.Tensor, enc: torch.Tensor, enc_mask: torch.Tensor, q_mask: torch.Tensor) -> torch.Tensor:
        from flash_attn import flash_attn_varlen_func  # type: ignore[import-untyped]
        from flash_attn.bert_padding import (  # type: ignore[import-untyped]
            index_first_axis,
            pad_input,
        )
        b, tq, h, d = q.shape
        tk = enc.shape[1]
        qm, km = q_mask.bool(), enc_mask.bool()
        q_idx = torch.nonzero(qm.flatten(), as_tuple=False).flatten()
        k_idx = torch.nonzero(km.flatten(), as_tuple=False).flatten()
        q_u = index_first_axis(q.reshape(b * tq, h, d), q_idx)
        enc_u = index_first_axis(enc.reshape(b * tk, -1), k_idx)
        k_u = self.k_norm(self.k_proj(enc_u).view(-1, self.num_kv, d))
        v_u = self.v_proj(enc_u).view(-1, self.num_kv, d)
        q_lens, k_lens = qm.sum(1, dtype=torch.int32), km.sum(1, dtype=torch.int32)
        cu_q = F.pad(torch.cumsum(q_lens, 0, dtype=torch.int32), (1, 0))
        cu_k = F.pad(torch.cumsum(k_lens, 0, dtype=torch.int32), (1, 0))
        out = flash_attn_varlen_func(q_u, k_u, v_u, cu_q, cu_k, int(q_lens.max()), int(k_lens.max()),
                                     softmax_scale=self.scaling, causal=False)
        return pad_input(out, q_idx, b, tq)  # [B, tq, H, d]; padded query rows come back as zeros


def flash_varlen_available(x: torch.Tensor) -> bool:
    if not (x.is_cuda and x.dtype in (torch.bfloat16, torch.float16)):
        return False
    try:
        import flash_attn  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        return False
    return True


class HybridLayer(nn.Module):
    """A Gemma 3 decoder layer with a cross-attention sublayer after self-attention. Called by the decoder's
    own forward with the layer's usual arguments; the encoder context comes from `set_context`."""

    def __init__(self, layer, norm_cls, checkpointing: bool):
        super().__init__()
        self.inner = layer
        self.inner.gradient_checkpointing = False  # this wrapper checkpoints the whole layer instead
        cfg = layer.config
        self.cross = CrossAttention(layer.self_attn, cfg)
        dt = layer.input_layernorm.weight.dtype
        self.pre_cross_layernorm = norm_cls(cfg.hidden_size, eps=cfg.rms_norm_eps).to(dt)
        self.post_cross_layernorm = norm_cls(cfg.hidden_size, eps=cfg.rms_norm_eps).to(dt)
        with torch.no_grad():  # Gemma3RMSNorm scales by (1 + weight): -1 closes the gate, the sublayer adds nothing
            self.post_cross_layernorm.weight.fill_(-1.0)
        self.checkpointing = checkpointing
        self.enc: torch.Tensor | None = None
        self.enc_mask: torch.Tensor | None = None
        self.enc_kv: tuple[torch.Tensor, torch.Tensor] | None = None
        self.dec_mask: torch.Tensor | None = None  # training: the decoder's padding mask, for the varlen kernel
        self.collect = False  # accumulate the cross-attention's share of the residual stream (validation)
        self.stat = [0.0, 0.0, 0]  # sum ||cross output||, sum ||residual||, rows

    def set_context(self, enc: torch.Tensor | None, enc_mask: torch.Tensor | None, cache_kv: bool = False,
                    dec_mask: torch.Tensor | None = None) -> None:
        self.enc, self.enc_mask, self.dec_mask = enc, enc_mask, dec_mask
        self.enc_kv = self.cross.kv(enc) if cache_kv and enc is not None else None

    def _forward(self, hidden_states, position_embeddings, attention_mask, position_ids, **kwargs):
        layer = self.inner
        residual = hidden_states
        x = layer.input_layernorm(hidden_states)
        x, _ = layer.self_attn(hidden_states=x, position_embeddings=position_embeddings, attention_mask=attention_mask,
                               position_ids=position_ids, **kwargs)
        h = residual + layer.post_attention_layernorm(x)
        residual = h
        assert self.enc is not None and self.enc_mask is not None, "HybridLayer: set_context before the decoder forward"
        x = self.cross(self.pre_cross_layernorm(h), self.enc, self.enc_mask, self.enc_kv, self.dec_mask)
        x = self.post_cross_layernorm(x)
        if self.collect:
            with torch.no_grad():
                self.stat[0] += float(x.float().norm(dim=-1).sum()); self.stat[1] += float(residual.float().norm(dim=-1).sum())
                self.stat[2] += x.shape[0] * x.shape[1]
        h = residual + x
        residual = h
        x = layer.mlp(layer.pre_feedforward_layernorm(h))
        return residual + layer.post_feedforward_layernorm(x)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, position_ids=None,
                past_key_values=None, **kwargs):
        if self.checkpointing and self.training and past_key_values is None:
            return torch.utils.checkpoint.checkpoint(partial(self._forward, **kwargs), hidden_states, position_embeddings,
                                                     attention_mask, position_ids, use_reentrant=False)
        return self._forward(hidden_states, position_embeddings, attention_mask, position_ids,
                             past_key_values=past_key_values, **kwargs)


class Hybrid(nn.Module):
    def __init__(self, dec_lm, text_encoder, dec_base: str, enc_base: str, attn: str, gradient_checkpointing: bool = True):
        """dec_lm: a causal LM (Gemma 3 family: `.model` with `.layers`, `.lm_head`); text_encoder: a bidirectional
        encoder whose forward takes input_ids and attention_mask and returns last_hidden_state. Prefer
        `Hybrid.from_pretrained`; this constructor takes built modules so a tiny pair can be tested on CPU."""
        super().__init__()
        from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
        self.dec_base, self.enc_base, self.attn = dec_base, enc_base, attn
        self.config = dec_lm.config
        dec_lm.config.use_cache = False
        self.decoder = dec_lm.model
        self.lm_head = dec_lm.lm_head
        self.encoder = text_encoder
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.decoder.gradient_checkpointing = False
        for i, layer in enumerate(self.decoder.layers):
            self.decoder.layers[i] = HybridLayer(layer, Gemma3RMSNorm, gradient_checkpointing)

    @classmethod
    def from_pretrained(cls, dec_base: str, enc_base: str, attn: str = "flash_attention_2", gradient_checkpointing: bool = True) -> "Hybrid":
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
        dec_lm = AutoModelForCausalLM.from_pretrained(dec_base, dtype=torch.bfloat16, attn_implementation=attn)
        full = AutoModelForSeq2SeqLM.from_pretrained(enc_base, dtype=torch.bfloat16, attn_implementation="sdpa")
        enc = full.model.encoder.text_model  # the rest (decoder, image tower, projector) is dropped
        del full
        return cls(dec_lm, enc, dec_base, enc_base, attn, gradient_checkpointing)

    # ---- pieces
    @property
    def layers(self) -> list[HybridLayer]:
        return list(self.decoder.layers)

    def encode(self, enc_ids: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids=enc_ids, attention_mask=enc_mask).last_hidden_state

    def set_context(self, enc: torch.Tensor | None, enc_mask: torch.Tensor | None, cache_kv: bool = False,
                    dec_mask: torch.Tensor | None = None) -> None:
        for layer in self.layers:
            layer.set_context(enc, enc_mask, cache_kv, dec_mask)

    def decode(self, dec_ids: torch.Tensor, dec_mask: torch.Tensor | None, position_ids: torch.Tensor | None = None,
               past=None) -> torch.Tensor:
        """Decoder hidden states after the final norm; the context must be set. With `past`, one decoding step."""
        return self.decoder(input_ids=dec_ids, attention_mask=dec_mask, position_ids=position_ids, past_key_values=past,
                            use_cache=past is not None).last_hidden_state

    def forward(self, enc_ids, enc_mask, dec_ids, dec_mask) -> torch.Tensor:
        """Decoder hidden states [B, T, H]; the caller applies lm_head where labels exist."""
        enc = self.encode(enc_ids, enc_mask)
        # the context stays set after the forward: under activation checkpointing the layers run again during the
        # backward and read it then; the next forward replaces it
        self.set_context(enc, enc_mask, dec_mask=dec_mask)
        return self.decode(dec_ids, dec_mask)

    # ---- diagnostics
    def collect_cross_stats(self, on: bool) -> None:
        for layer in self.layers:
            layer.collect = on
            if on:
                layer.stat = [0.0, 0.0, 0]

    def cross_stats(self) -> dict[str, float]:
        """Mean over rows of ||cross-attention output|| / ||residual stream|| per layer, summarised: the mean over
        layers and the largest layer. Near zero means the decoder is not reading the encoder."""
        ratios = [layer.stat[0] / layer.stat[1] for layer in self.layers if layer.stat[1] > 0]
        if not ratios:
            return {}
        with torch.no_grad():  # the gate itself: mean |1 + w| per layer (0 closed; Gemma's own post-norms sit near 1)
            gates = [float((1.0 + layer.post_cross_layernorm.weight.float()).abs().mean()) for layer in self.layers]
        return {"cross_ratio_mean": sum(ratios) / len(ratios), "cross_ratio_max": max(ratios),
                "cross_gate_mean": sum(gates) / len(gates), "cross_gate_max": max(gates)}

    # ---- persistence
    def save(self, out: Path, tokenizer=None) -> None:
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out / "hybrid.pt")
        (out / "hybrid.json").write_text(json.dumps({"dec_base": self.dec_base, "enc_base": self.enc_base, "attn": self.attn}))
        if tokenizer is not None:
            tokenizer.save_pretrained(out)

    @classmethod
    def load(cls, path: str | Path, attn: str | None = None, gradient_checkpointing: bool = True) -> "Hybrid":
        path = Path(path)
        meta = json.loads((path / "hybrid.json").read_text())
        m = cls.from_pretrained(meta["dec_base"], meta["enc_base"], attn or meta["attn"], gradient_checkpointing)
        m.load_state_dict(torch.load(path / "hybrid.pt", map_location="cpu"))
        return m

    @staticmethod
    def is_hybrid_dir(path: str | Path) -> bool:
        return (Path(path) / "hybrid.json").exists()
