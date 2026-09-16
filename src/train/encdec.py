"""E3: an encoder-decoder built from one Qwen3 decoder checkpoint (the T5Gemma recipe).

Encoder: the base model's layers with bidirectional attention over the real tokens (FlashAttention 2
runs non-causally when the attention modules say so). Decoder: the same layers again, causal self-
attention over the target, plus a cross-attention sublayer between self-attention and the MLP in every
layer (T5 ordering). Cross-attention queries, keys, values and the per-head q/k norms are copied from
that layer's self-attention; the output projection starts at zero, so at initialisation the decoder is
exactly the causal base model and learns to read the encoder during adaptation. No rotary embedding on
cross-attention. Output head tied to the decoder embeddings. Decoder start token: `<|endoftext|>`.

Checkpoint layout (`EncDec.save` / `EncDec.load`): `encdec.pt` (state dict), `encdec.json` (base id
and attention implementation), tokenizer files.
"""

import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm


class CrossAttention(nn.Module):
    def __init__(self, self_attn, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv = config.num_key_value_heads
        self.head_dim = self_attn.head_dim
        self.q_proj = copy.deepcopy(self_attn.q_proj)
        self.k_proj = copy.deepcopy(self_attn.k_proj)
        self.v_proj = copy.deepcopy(self_attn.v_proj)
        self.q_norm = copy.deepcopy(self_attn.q_norm)
        self.k_norm = copy.deepcopy(self_attn.k_norm)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False,
                                dtype=self_attn.o_proj.weight.dtype, device=self_attn.o_proj.weight.device)
        nn.init.zeros_(self.o_proj.weight)

    def forward(self, hidden: torch.Tensor, enc: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
        b, tq, _ = hidden.shape
        tk = enc.shape[1]
        q = self.q_norm(self.q_proj(hidden).view(b, tq, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(enc).view(b, tk, self.num_kv, self.head_dim)).transpose(1, 2)
        v = self.v_proj(enc).view(b, tk, self.num_kv, self.head_dim).transpose(1, 2)
        mask = enc_mask.bool()[:, None, None, :]  # keys at padding are never attended
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        return self.o_proj(out.transpose(1, 2).reshape(b, tq, self.num_heads * self.head_dim))


class LoRALinear(nn.Module):
    """y = W x + (B A x) * (alpha / r); B starts at zero so the wrapped layer is unchanged at initialisation."""

    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=base.weight.dtype, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=base.weight.dtype, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def inject_lora(tower, r: int, alpha: int) -> int:
    """Wrap every attention and MLP projection in a Qwen3 tower with a LoRA adapter; returns how many."""
    n = 0
    for layer in tower.layers:  # a Qwen3Model
        for parent in (layer.self_attn, layer.mlp):
            for name in LORA_TARGETS:
                lin = getattr(parent, name, None)
                if isinstance(lin, nn.Linear):
                    setattr(parent, name, LoRALinear(lin, r, alpha))
                    n += 1
    return n


class EncDec(nn.Module):
    def __init__(self, base: str, attn: str = "flash_attention_2", gradient_checkpointing: bool = True,
                 lora_r: int = 0, lora_alpha: int = 0):
        super().__init__()
        enc_lm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation=attn)
        dec_lm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation=attn)
        for lm in (enc_lm, dec_lm):
            lm.config.use_cache = False
            if gradient_checkpointing:
                lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.config = dec_lm.config
        self.base, self.attn = base, attn
        self.encoder = enc_lm.model
        for layer in self.encoder.layers:
            layer.self_attn.is_causal = False
        self.decoder = dec_lm.model
        self.lm_head = dec_lm.lm_head
        self.cross = nn.ModuleList([CrossAttention(layer.self_attn, self.config) for layer in self.decoder.layers])
        self.cross_norm = nn.ModuleList([Qwen3RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps).to(torch.bfloat16)
                                         for _ in self.decoder.layers])
        self.checkpointing = gradient_checkpointing
        self.lora_r, self.lora_alpha = lora_r, lora_alpha
        if lora_r:
            # the two towers train through LoRA (3.8B parameters do not fit fp32 master weights on a 32 GB
            # card); the cross-attention sublayers and every norm train in full; embeddings and head frozen
            inject_lora(self.encoder, lora_r, lora_alpha or 2 * lora_r)
            inject_lora(self.decoder, lora_r, lora_alpha or 2 * lora_r)
            for n, p in self.named_parameters():
                trainable = (".A" in n and n.endswith(".A")) or n.endswith(".B") or n.startswith("cross") or "norm" in n
                p.requires_grad_(trainable)

    # ---- forward pieces
    def encode(self, enc_ids: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids=enc_ids, attention_mask=enc_mask).last_hidden_state

    def _layer(self, i: int, h, pos_emb, mask, enc, enc_mask, past, cache_position, position_ids):
        layer = self.decoder.layers[i]
        res = h
        x = layer.input_layernorm(h)
        x, _ = layer.self_attn(hidden_states=x, position_embeddings=pos_emb, attention_mask=mask, past_key_values=past,
                               cache_position=cache_position, position_ids=position_ids, use_cache=past is not None)
        h = res + x
        res = h
        x = self.cross[i](self.cross_norm[i](h), enc, enc_mask)
        h = res + x
        res = h
        x = layer.mlp(layer.post_attention_layernorm(h))
        return res + x

    def decode(self, dec_ids: torch.Tensor, dec_mask: torch.Tensor | None, enc: torch.Tensor, enc_mask: torch.Tensor,
               past=None, cache_position: torch.Tensor | None = None) -> torch.Tensor:
        h = self.decoder.embed_tokens(dec_ids)
        past_len = past.get_seq_length() if past is not None else 0
        if cache_position is None:
            cache_position = torch.arange(past_len, past_len + dec_ids.shape[1], device=dec_ids.device)
        position_ids = cache_position[None]
        pos_emb = self.decoder.rotary_emb(h, position_ids)
        mask = create_causal_mask(config=self.config, inputs_embeds=h, attention_mask=dec_mask, past_key_values=past,
                                  position_ids=position_ids)
        for i in range(len(self.decoder.layers)):
            if self.checkpointing and self.training and past is None:
                h = torch.utils.checkpoint.checkpoint(self._layer, i, h, pos_emb, mask, enc, enc_mask, None, cache_position,
                                                      position_ids, use_reentrant=False)
            else:
                h = self._layer(i, h, pos_emb, mask, enc, enc_mask, past, cache_position, position_ids)
        return self.decoder.norm(h)

    def forward(self, enc_ids, enc_mask, dec_ids, dec_mask) -> tuple[torch.Tensor, torch.Tensor]:
        """(decoder hidden states [B, T, H], encoder states [B, S, H]); the caller applies lm_head where
        labels exist, on either side."""
        enc = self.encode(enc_ids, enc_mask)
        return self.decode(dec_ids, dec_mask, enc, enc_mask), enc

    # ---- persistence
    def save(self, out: Path, tokenizer=None) -> None:
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out / "encdec.pt")
        (out / "encdec.json").write_text(json.dumps({"base": self.base, "attn": self.attn, "lora_r": self.lora_r, "lora_alpha": self.lora_alpha}))
        if tokenizer is not None:
            tokenizer.save_pretrained(out)

    @classmethod
    def load(cls, path: str | Path, attn: str | None = None, gradient_checkpointing: bool = True) -> "EncDec":
        path = Path(path)
        meta = json.loads((path / "encdec.json").read_text())
        m = cls(meta["base"], attn or meta["attn"], gradient_checkpointing, meta.get("lora_r", 0), meta.get("lora_alpha", 0))
        m.load_state_dict(torch.load(path / "encdec.pt", map_location="cpu"))
        return m

    @staticmethod
    def is_encdec_dir(path: str | Path) -> bool:
        return (Path(path) / "encdec.json").exists()


def load_tokenizer(base_or_dir: str):
    return AutoTokenizer.from_pretrained(base_or_dir)
