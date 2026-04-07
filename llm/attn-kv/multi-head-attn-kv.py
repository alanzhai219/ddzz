import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


class MultiHeadAttentionWithKVCache(nn.Module):
    """Minimal multi-head self-attention (inference only) with KV cache.

    Input:
        x: [B, T, D] where T is the current chunk length.
        cache:
            - k: [B, H, T_cache, Hd]
            - v: [B, H, T_cache, Hd]

    Output:
        out: [B, T, D]
        new_cache (optional): updated cache after appending current K/V.
    """

    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # Wq: [D, D], bq: [D]
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # Wk: [D, D], bk: [D]
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # Wv: [D, D], bv: [D]
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, D] -> [B, H, T, Hd]
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, H, T, Hd] -> [B, T, D]
        b, _, t, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, self.embed_dim)

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if x.dim() != 3:
            raise ValueError("x must be [B, T, D]")

        b, t, d = x.shape
        if d != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, but got {d}")

        q = self._split_heads(self.q_proj(x))  # [B, H, T, Hd]
        k_new = self._split_heads(self.k_proj(x))  # [B, H, T, Hd]
        v_new = self._split_heads(self.v_proj(x))  # [B, H, T, Hd]

        past_k = None
        past_v = None
        if cache is not None:
            past_k = cache.get("k")
            past_v = cache.get("v")

        if past_k is not None:
            if past_k.shape != past_v.shape:
                raise ValueError("cache k/v shape mismatch")
            k = torch.cat([past_k, k_new], dim=2)
            v = torch.cat([past_v, v_new], dim=2)
        else:
            k = k_new
            v = v_new

        # [B, H, T, Hd] x [B, H, Hd, T_total] -> [B, H, T, T_total]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        attn = torch.softmax(scores, dim=-1)

        # [B, H, T, T_total] x [B, H, T_total, Hd] -> [B, H, T, Hd]
        context = torch.matmul(attn, v)
        context = self._merge_heads(context)  # [B, T, D]

        out = self.out_proj(context)

        if not use_cache:
            return out

        new_cache: Dict[str, torch.Tensor] = {"k": k, "v": v}
        return out, new_cache


if __name__ == "__main__":
    torch.manual_seed(0)

    b, d, h = 2, 64, 8
    x_prefill = torch.randn(b, 5, d)
    x_decode = torch.randn(b, 1, d)

    attn = MultiHeadAttentionWithKVCache(embed_dim=d, num_heads=h)
    attn.eval()

    y_prefill, cache = attn(
        x_prefill,
        use_cache=True,
    )
    y_decode, cache = attn(
        x_decode,
        cache=cache,
        use_cache=True,
    )

    print("prefill output:", y_prefill.shape)
    print("decode output:", y_decode.shape)
    print("cache k:", cache["k"].shape)
    print("cache v:", cache["v"].shape)
