import math

import torch
import torch.nn as nn


class MultiHeadAttentionInference(nn.Module):
    """Classic multi-head self-attention for inference only.

    Input:
        x: [B, S, D]

    Output:
        y: [B, S, D]
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
    ) -> None:
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
        # [B, S, D] -> [B, H, S, Hd]
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, H, S, Hd] -> [B, S, D]
        b, _, s, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, s, self.embed_dim)

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute classic multi-head self-attention in inference mode."""
        if x.dim() != 3:
            raise ValueError("x must be [B, S, D]")
        b, s, d = x.shape
        if d != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, but got {d}")

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # [B, H, S, Hd] x [B, H, Hd, S] -> [B, H, S, S]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        attn = torch.softmax(scores, dim=-1)

        # [B, H, S, S] x [B, H, S, Hd] -> [B, H, S, Hd]
        context = torch.matmul(attn, v)
        context = self._merge_heads(context)

        out = self.out_proj(context)

        return out


if __name__ == "__main__":
    torch.manual_seed(0)

    x = torch.randn(2, 8, 64)

    attn = MultiHeadAttentionInference(embed_dim=64, num_heads=8)
    attn.eval()

    y = attn(x)
    print("input:", x.shape)
    print("output:", y.shape)
