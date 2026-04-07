import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


class SingleHeadAttentionWithKVCache(nn.Module):
    """Minimal single-head self-attention (inference only) with KV cache.

    Input:
        x: [B, T, D] where T is the current chunk length.
        cache:
            - k: [B, T_cache, D]
            - v: [B, T_cache, D]

    Output:
        out: [B, T, D]
        new_cache (optional): updated cache after appending current K/V.
    """

    def __init__(self, embed_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # Wq: [D, D], bq: [D]
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # Wk: [D, D], bk: [D]
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # Wv: [D, D], bv: [D]
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

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

        q = self.q_proj(x)  # [B, T, D]
        k_new = self.k_proj(x)  # [B, T, D]
        v_new = self.v_proj(x)  # [B, T, D]

        past_k = None
        past_v = None
        if cache is not None:
            past_k = cache.get("k")
            past_v = cache.get("v")

        if past_k is not None:
            if past_k.shape[:2] != past_v.shape[:2]:
                raise ValueError("cache k/v length mismatch")
            k = torch.cat([past_k, k_new], dim=1)
            v = torch.cat([past_v, v_new], dim=1)
        else:
            k = k_new
            v = v_new

        # [B, T, D] x [B, D, T_total] -> [B, T, T_total]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim)

        attn = torch.softmax(scores, dim=-1)

        # [B, T, T_total] x [B, T_total, D] -> [B, T, D]
        context = torch.matmul(attn, v)
        out = self.out_proj(context)

        if not use_cache:
            return out

        new_cache: Dict[str, torch.Tensor] = {"k": k, "v": v}
        return out, new_cache


if __name__ == "__main__":
    torch.manual_seed(0)

    b, d = 2, 64
    x_prefill = torch.randn(b, 5, d)
    x_decode = torch.randn(b, 1, d)

    attn = SingleHeadAttentionWithKVCache(embed_dim=d)
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
