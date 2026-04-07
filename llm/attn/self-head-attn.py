import math

import torch
import torch.nn as nn


class SingleHeadAttentionInference(nn.Module):
    """Single-head self-attention for inference only.

    Input shape:
        x: [batch_size, seq_len, embed_dim]

    Output shape:
        y: [batch_size, seq_len, embed_dim]
    """

    def __init__(
        self,
        embed_dim: int,
        bias: bool = True,
    ) -> None:
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
    ) -> torch.Tensor:
        """Compute classic single-head self-attention for inference."""
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, S, D] x [B, D, S] -> [B, S, S]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim)

        attn = torch.softmax(scores, dim=-1)

        # [B, S, S] x [B, S, D] -> [B, S, D]
        context = torch.matmul(attn, v)

        out = self.out_proj(context)

        return out


if __name__ == "__main__":
    x = torch.randn(2, 8, 64)
    attn = SingleHeadAttentionInference(embed_dim=64)
    attn.eval()
    y = attn(x)
    print("input:", x.shape)
    print("output:", y.shape)
