"""E5: an encoder stitched into a different decoder's embedding space (soft tokens).

The decoder is a causal language model exactly as E1 fine-tuned it (Gemma 3 1B): no new attention weights, its own
self-attention over the whole sequence, loss on the answer only. What changes is its input at the prompt positions:
instead of the token embeddings, the decoder reads the bidirectional text encoder of a native encoder-decoder from the
same family (T5Gemma 2) run over the prompt, one state per prompt token, mapped into the decoder's embedding space by
an affine stitch. The answer tokens enter through the decoder's ordinary embeddings. The two checkpoints share the
tokenizer, so the encoder's positions and the decoder's prompt positions correspond one to one and the decoder sees
exactly as many positions as in E1.

The stitch is fitted beforehand by ridge regression from the encoder's states onto the decoder's embedding of the
token each state came from (`src/train/fit_stitch.py --target-embeddings`), so that at step 0 every stitched vector
resembles what E1's decoder would have read at that position; everything trains afterwards. Gemma scales its token
embeddings by the square root of the width inside the embedding module and vectors passed in directly skip that, so
the regression target, and the answer-token path here, is the scaled embedding.

Same batches as the E6 decoder-only + encoder arm (`collate_encdec(dec_prompt=True, dec_start=False)`: encoder side
the prompt, decoder side the prompt then the target, labels on the target).

Checkpoint layout (`SoftStitched.save` / `SoftStitched.load`): `soft.pt` (state dict), `soft.json` (decoder base,
encoder base, attention implementation), tokenizer files.
"""

import json
from pathlib import Path

import torch
from torch import nn


class SoftStitched(nn.Module):
    def __init__(self, dec_lm, text_encoder, dec_base: str, enc_base: str, attn: str, gradient_checkpointing: bool = True,
                 stitch: nn.Linear | None = None):
        """dec_lm: a causal LM (`.model` with `.embed_tokens` and `.layers`, `.lm_head`); text_encoder: a bidirectional
        encoder whose forward takes input_ids and attention_mask and returns last_hidden_state. Built modules, so a tiny
        pair can be tested on CPU; prefer `SoftStitched.from_pretrained`."""
        super().__init__()
        self.dec_base, self.enc_base, self.attn = dec_base, enc_base, attn
        # a multimodal checkpoint (Gemma 3 4B and up) keeps its text decoder under `language_model`; the image tower and
        # projector are never called and are not kept
        text = dec_lm.model.language_model if hasattr(dec_lm.model, "language_model") else dec_lm.model
        self.config = text.config  # the text decoder's: what generation's cache and the loop's MFU term read
        self.config.use_cache = False
        self.decoder = text
        self.lm_head = dec_lm.lm_head
        self.encoder = text_encoder
        d_enc, d_dec = text_encoder.config.hidden_size, text.config.hidden_size
        dt = next(self.decoder.parameters()).dtype
        self.stitch = stitch if stitch is not None else nn.Linear(d_enc, d_dec, bias=True, dtype=dt)
        if gradient_checkpointing:
            for tower in (self.encoder, self.decoder):
                tower.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    @classmethod
    def from_pretrained(cls, dec_base: str, enc_base: str, attn: str = "flash_attention_2", gradient_checkpointing: bool = True,
                        stitch_path: str | Path | None = None) -> "SoftStitched":
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
        dec_lm = AutoModelForCausalLM.from_pretrained(dec_base, dtype=torch.bfloat16, attn_implementation=attn)
        full = AutoModelForSeq2SeqLM.from_pretrained(enc_base, dtype=torch.bfloat16, attn_implementation="sdpa")
        enc = full.model.encoder.text_model  # the rest (decoder, image tower, projector) is dropped
        del full
        stitch = None
        if stitch_path is not None:
            fit = torch.load(Path(stitch_path), map_location="cpu")
            stitch = nn.Linear(fit["weight"].shape[1], fit["weight"].shape[0], bias=True, dtype=torch.bfloat16)
            with torch.no_grad():
                stitch.weight.copy_(fit["weight"])
                stitch.bias.copy_(fit["bias"])
        return cls(dec_lm, enc, dec_base, enc_base, attn, gradient_checkpointing, stitch)

    # ---- pieces
    def encode(self, enc_ids: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
        """Stitched encoder states [B, S, d_dec]: what the decoder reads at the prompt positions."""
        return self.stitch(self.encoder(input_ids=enc_ids, attention_mask=enc_mask).last_hidden_state)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """The decoder's own (scaled) token embeddings: the answer path, and the regression target of the stitch."""
        return self.decoder.embed_tokens(ids)

    def inputs(self, enc: torch.Tensor, enc_mask: torch.Tensor, dec_ids: torch.Tensor) -> torch.Tensor:
        """Decoder input embeddings [B, T, d_dec]: the stitched states where the decoder side carries the prompt (its
        first positions, right-padded as the encoder side), the token embeddings everywhere else."""
        emb = self.embed(dec_ids)
        s = min(enc.shape[1], emb.shape[1])
        where = torch.zeros(emb.shape[:2], dtype=torch.bool, device=emb.device)
        where[:, :s] = enc_mask[:, :s].bool()
        soft = torch.zeros_like(emb)
        soft[:, :s] = enc[:, :s].to(emb.dtype)
        return torch.where(where[..., None], soft, emb)

    def decode(self, inputs_embeds: torch.Tensor, dec_mask: torch.Tensor | None, position_ids: torch.Tensor | None = None,
               past=None) -> torch.Tensor:
        """Decoder hidden states after the final norm. With `past`, one decoding step."""
        return self.decoder(inputs_embeds=inputs_embeds, attention_mask=dec_mask, position_ids=position_ids,
                            past_key_values=past, use_cache=past is not None).last_hidden_state

    def forward(self, enc_ids, enc_mask, dec_ids, dec_mask) -> torch.Tensor:
        """Decoder hidden states [B, T, d_dec]; the caller applies lm_head where labels exist."""
        return self.decode(self.inputs(self.encode(enc_ids, enc_mask), enc_mask, dec_ids), dec_mask)

    # ---- persistence
    def save(self, out: Path, tokenizer=None) -> None:
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out / "soft.pt")
        (out / "soft.json").write_text(json.dumps({"dec_base": self.dec_base, "enc_base": self.enc_base, "attn": self.attn}))
        if tokenizer is not None:
            tokenizer.save_pretrained(out)

    @classmethod
    def load(cls, path: str | Path, attn: str | None = None, gradient_checkpointing: bool = True) -> "SoftStitched":
        path = Path(path)
        meta = json.loads((path / "soft.json").read_text())
        m = cls.from_pretrained(meta["dec_base"], meta["enc_base"], attn or meta["attn"], gradient_checkpointing)
        sd = torch.load(path / "soft.pt", map_location="cpu")
        # the head is tied to the decoder's embeddings; an export gathered from FSDP shards carries the shared tensor
        # once, under the embedding key, so the head key may be absent: load what is there and re-tie
        missing, unexpected = m.load_state_dict(sd, strict=False)
        tied = {"lm_head.weight"}
        if unexpected or set(missing) - tied:
            raise RuntimeError(f"soft checkpoint mismatch: missing {sorted(set(missing) - tied)}, unexpected {sorted(unexpected)}")
        if missing:
            m.lm_head.weight = m.decoder.embed_tokens.weight
        return m

    @staticmethod
    def is_soft_dir(path: str | Path) -> bool:
        return (Path(path) / "soft.json").exists()
