"""Attention masks for the prefix-LM arm (E2): every prompt token attends to every prompt token,
answer tokens attend causally to everything before them, padding is never attended and a padding
query still sees itself (a fully masked row would produce NaN in the softmax).

Two encodings of the same rule: a FlexAttention BlockMask (block-sparse, the fast path) and a dense
boolean [B, 1, Q, K] mask for SDPA (the fallback; True = may attend). Both are passed to the model as
`attention_mask={"full_attention": mask}`, which the Qwen3 forward uses as-is instead of building a
causal mask (transformers 5, modeling_qwen3.Qwen3Model.forward).
"""

import torch


def prefix_lm_allowed(prompt_len: torch.Tensor, seq_len: torch.Tensor):
    """mask_mod(b, h, q, kv) -> bool for FlexAttention; also the rule the dense mask materialises."""

    def allowed(b, h, q, kv):
        return ((kv < prompt_len[b]) | (kv <= q)) & ((kv < seq_len[b]) | (kv == q))

    return allowed


def dense_prefix_lm_mask(prompt_len: torch.Tensor, seq_len: torch.Tensor, n: int) -> torch.Tensor:
    """[B, 1, n, n] boolean mask; prompt_len and seq_len are per-row tensors on the target device."""
    q = torch.arange(n, device=prompt_len.device)
    kv = q
    pl = prompt_len[:, None, None]
    sl = seq_len[:, None, None]
    m = ((kv[None, None, :] < pl) | (kv[None, None, :] <= q[None, :, None])) & ((kv[None, None, :] < sl) | (kv[None, None, :] == q[None, :, None]))
    return m[:, None]


def block_prefix_lm_mask(prompt_len: torch.Tensor, seq_len: torch.Tensor, n: int):
    from torch.nn.attention.flex_attention import create_block_mask

    return create_block_mask(prefix_lm_allowed(prompt_len, seq_len), B=prompt_len.shape[0], H=None, Q_LEN=n, KV_LEN=n,
                             device=prompt_len.device, _compile=True)


def prefix_lm_mask(prompt_len: torch.Tensor, seq_len: torch.Tensor, n: int, attn: str):
    """The mask in the encoding the attention implementation wants: BlockMask for flex_attention, dense
    boolean for sdpa. flash_attention_2 cannot express it."""
    if attn == "flex_attention":
        return block_prefix_lm_mask(prompt_len, seq_len, n)
    if attn == "sdpa":
        return dense_prefix_lm_mask(prompt_len, seq_len, n)
    raise ValueError(f"prefix-LM needs flex_attention or sdpa, not {attn}")


# FlexAttention block sizes that fit the 101 KB shared memory of consumer Blackwell (RTX 5090); the
# default autotune configs need 120 KB and the backward fails to compile ("No valid triton configs").
FLEX_KERNEL_OPTIONS = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32}


def left_padded_prompt_mask(attn2d: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """[B, 1, n, n] boolean mask for a left-padded batch of prompts at prefill: real tokens attend to
    every real token (bidirectional, the prefix-LM prefill) or to the real tokens before them
    (causal=True, for checking the decode loop against a causal model); padding rows see themselves."""
    n = attn2d.shape[1]
    real = attn2d.bool()
    q = torch.arange(n, device=attn2d.device)
    allowed = real[:, None, :].expand(-1, n, -1).clone()
    if causal:
        allowed &= (q[None, :] <= q[:, None])[None]
    allowed |= torch.eye(n, dtype=torch.bool, device=attn2d.device)[None]
    return allowed[:, None]


def left_padded_block_mask(attn2d: torch.Tensor):
    """FlexAttention BlockMask for a left-padded prefill where every real token attends to every real
    token (the prefix-LM prefill: the whole prompt is prefix); padding rows see themselves."""
    from torch.nn.attention.flex_attention import create_block_mask

    n = attn2d.shape[1]
    left = n - attn2d.sum(1)

    def allowed(b, h, q, kv):
        return (kv >= left[b]) | (kv == q)

    return create_block_mask(allowed, B=attn2d.shape[0], H=None, Q_LEN=n, KV_LEN=n, device=attn2d.device, _compile=True)
