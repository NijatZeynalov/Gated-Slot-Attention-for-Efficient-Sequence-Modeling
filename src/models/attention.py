import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GatedSlotAttention(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            num_slots: int,
            dropout: float = 0.1
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.gate = nn.Linear(hidden_size, num_slots)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        scores = (q @ k.transpose(-2, -1)) * self.scaling
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Gating
        gates = F.softmax(self.gate(x), dim=-1)

        # Apply attention and gating
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x * gates.unsqueeze(-1)

        return self.out_proj(x), gates