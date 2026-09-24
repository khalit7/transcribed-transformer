"""E3-mix-and-match: a T5Gemma 2 encoder of one size with a T5Gemma 2 decoder of another, joined by a linear stitch.

The decoder reads encoder states through its own key and value projections (merged attention, no separate
cross-attention weights), so the states it receives must have the decoder's width and, to be readable from
step 0, roughly the distribution of its own encoder's states. The stitch is one affine map on the encoder's
final states, before the decoder, fitted beforehand by ridge regression from the donor encoder's states onto
the target-side encoder's states over the same tokens (`src/train/fit_stitch.py`); it then trains with
everything else. Imitation of the smaller encoder is the starting point, not the goal: the fine-tune moves the
map, the encoder and the decoder to whatever the task rewards.

Checkpoint layout (`Stitched.save` / `Stitched.load`): `stitched.pt` (state dict), `stitched.json` (encoder
base, decoder base, attention implementation), tokenizer files.
"""

import json
from pathlib import Path

import torch
from torch import nn


class Stitched(nn.Module):
    def __init__(self, text_encoder, decoder, lm_head, enc_base: str, dec_base: str, attn: str,
                 gradient_checkpointing: bool = True, stitch: nn.Linear | None = None):
        """text_encoder: a T5Gemma2TextEncoder (final states after its norm); decoder: a T5Gemma2Decoder taking
        encoder_hidden_states of its own width; lm_head: the decoder's head. Built modules, so a tiny pair can be
        tested on CPU; prefer `Stitched.from_pretrained`."""
        super().__init__()
        self.enc_base, self.dec_base, self.attn = enc_base, dec_base, attn
        self.encoder = text_encoder
        self.decoder = decoder
        self.lm_head = lm_head
        self.config = decoder.config  # the decoder's: what generation's cache and the loop's MFU term read
        d_enc, d_dec = text_encoder.config.hidden_size, decoder.config.hidden_size
        dt = next(decoder.parameters()).dtype
        self.stitch = stitch if stitch is not None else nn.Linear(d_enc, d_dec, bias=True, dtype=dt)
        if gradient_checkpointing:
            for tower in (self.encoder, self.decoder):
                tower.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.decoder.config.use_cache = False

    @classmethod
    def from_pretrained(cls, enc_base: str, dec_base: str, attn: str = "sdpa", gradient_checkpointing: bool = True,
                        stitch_path: str | Path | None = None) -> "Stitched":
        from transformers import AutoModelForSeq2SeqLM
        enc_full = AutoModelForSeq2SeqLM.from_pretrained(enc_base, dtype=torch.bfloat16, attn_implementation=attn)
        dec_full = AutoModelForSeq2SeqLM.from_pretrained(dec_base, dtype=torch.bfloat16, attn_implementation=attn)
        enc = enc_full.model.encoder.text_model  # the image tower, projector and the donor's decoder are dropped
        dec, head = dec_full.model.decoder, dec_full.lm_head  # the donor's encoder is dropped; head stays tied to the embeddings
        del enc_full, dec_full
        stitch = None
        if stitch_path is not None:
            fit = torch.load(Path(stitch_path), map_location="cpu")
            stitch = nn.Linear(fit["weight"].shape[1], fit["weight"].shape[0], bias=True, dtype=torch.bfloat16)
            with torch.no_grad():
                stitch.weight.copy_(fit["weight"])
                stitch.bias.copy_(fit["bias"])
        return cls(enc, dec, head, enc_base, dec_base, attn, gradient_checkpointing, stitch)

    # ---- pieces
    def encode(self, enc_ids: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
        """Stitched encoder states [B, S, d_dec]: what the decoder reads, and what a per-case cache would hold."""
        return self.stitch(self.encoder(input_ids=enc_ids, attention_mask=enc_mask).last_hidden_state)

    def decode(self, dec_ids: torch.Tensor, dec_mask: torch.Tensor | None, enc: torch.Tensor, enc_mask: torch.Tensor,
               position_ids: torch.Tensor | None = None, past=None) -> torch.Tensor:
        return self.decoder(input_ids=dec_ids, attention_mask=dec_mask, position_ids=position_ids, past_key_values=past,
                            encoder_hidden_states=enc, encoder_attention_mask=enc_mask, use_cache=past is not None,
                            return_dict=True).last_hidden_state

    def forward(self, enc_ids, enc_mask, dec_ids, dec_mask) -> torch.Tensor:
        """Decoder hidden states [B, T, d_dec]; the caller applies lm_head where labels exist."""
        return self.decode(dec_ids, dec_mask, self.encode(enc_ids, enc_mask), enc_mask)

    # ---- persistence
    def save(self, out: Path, tokenizer=None) -> None:
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out / "stitched.pt")
        (out / "stitched.json").write_text(json.dumps({"enc_base": self.enc_base, "dec_base": self.dec_base, "attn": self.attn}))
        if tokenizer is not None:
            tokenizer.save_pretrained(out)

    @classmethod
    def load(cls, path: str | Path, attn: str | None = None, gradient_checkpointing: bool = True) -> "Stitched":
        path = Path(path)
        meta = json.loads((path / "stitched.json").read_text())
        m = cls.from_pretrained(meta["enc_base"], meta["dec_base"], attn or meta["attn"], gradient_checkpointing)
        m.load_state_dict(torch.load(path / "stitched.pt", map_location="cpu"))
        return m

    @staticmethod
    def is_stitched_dir(path: str | Path) -> bool:
        return (Path(path) / "stitched.json").exists()


class AffineMoments:
    """Running moments for ridge regression of y on x with a bias: G = [x 1]ᵀ[x 1] and [x 1]ᵀy, accumulated chunk by
    chunk in float64 so millions of paired states never sit in memory at once."""

    def __init__(self, d_in: int, d_out: int, device=None):
        self.g = torch.zeros(d_in + 1, d_in + 1, dtype=torch.float64, device=device)
        self.xty = torch.zeros(d_in + 1, d_out, dtype=torch.float64, device=device)
        self.n = 0

    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        xb = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)], 1).to(self.g.device, torch.float64)
        self.g += xb.T @ xb
        self.xty += xb.T @ y.to(self.g.device, torch.float64)
        self.n += x.shape[0]

    def solve(self, ridge: float = 1e-2) -> tuple[torch.Tensor, torch.Tensor]:
        """(weight [d_out, d_in], bias [d_out]) for nn.Linear. The ridge is relative to the mean diagonal of the Gram
        matrix, so one value serves widths and scales alike; the bias is not shrunk."""
        lam = ridge * self.g.diagonal()[:-1].mean()
        reg = torch.eye(self.g.shape[0], dtype=self.g.dtype, device=self.g.device) * lam
        reg[-1, -1] = 0.0
        w = torch.linalg.solve(self.g + reg, self.xty)  # [d_in + 1, d_out]
        return w[:-1].T.contiguous(), w[-1].contiguous()


def fit_affine(x: torch.Tensor, y: torch.Tensor, ridge: float = 1e-2) -> tuple[torch.Tensor, torch.Tensor]:
    """Ridge regression of y on x with a bias in one go: x [N, d_in], y [N, d_out]. Returns (weight, bias) for nn.Linear."""
    m = AffineMoments(x.shape[1], y.shape[1], x.device)
    m.add(x, y)
    w, b = m.solve(ridge)
    return w.to(x.dtype), b.to(x.dtype)


def explained_variance(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> float:
    """1 - residual variance / total variance of y, over every element (held-out rows)."""
    w, b = weight.to(x.device, x.dtype), bias.to(x.device, x.dtype)
    pred = x @ w.T + b
    return float(1.0 - (y - pred).pow(2).sum() / (y - y.mean(0)).pow(2).sum())
