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
    def forward(self, x: torch.Tensor, use_sdpa: bool = False, is_causal: bool = False) -> torch.Tensor:
        """Compute classic single-head self-attention for inference."""

        # llm node: 3 matmuls for q, k, v
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # sdpa core formula:
        # step1: scores = q @ k^T / sqrt(d)
        # step2: attn = softmax(scores)
        # step3: context = attn @ v
        if use_sdpa:
            # F.scaled_dot_product_attention expects [B, ..., S, D]
            # For single-head, no extra head dim needed; it treats the last two dims as (S, D).
            context = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        else:
            # [B, S, D] x [B, D, S] -> [B, S, S]
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim)

            if is_causal:
                seq_len = scores.size(-1)
                causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)
                scores = scores + causal_mask

            attn = torch.softmax(scores, dim=-1)

            # [B, S, S] x [B, S, D] -> [B, S, D]
            context = torch.matmul(attn, v)

        out = self.out_proj(context)

        return out


if __name__ == "__main__":
    x = torch.randn(2, 8, 64)
    attn = SingleHeadAttentionInference(embed_dim=64)
    attn.eval()

    y_manual = attn(x, use_sdpa=False)
    y_sdpa = attn(x, use_sdpa=True)
    print("input:", x.shape)
    print("manual:", y_manual.shape)
    print("sdpa:  ", y_sdpa.shape)
    print("max diff (no causal):", (y_manual - y_sdpa).abs().max().item())

    y_manual_causal = attn(x, use_sdpa=False, is_causal=True)
    y_sdpa_causal = attn(x, use_sdpa=True, is_causal=True)
    print("max diff (causal):   ", (y_manual_causal - y_sdpa_causal).abs().max().item())
