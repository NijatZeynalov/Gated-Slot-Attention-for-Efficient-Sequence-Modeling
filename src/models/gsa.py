import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GSAConfig:
    hidden_size: int = 4096
    num_slots: int = 64
    slot_size: int = 256
    slot_dropout: float = 0.1
    gate_temperature: float = 1.0


class GSALayer(nn.Module):
    def __init__(self, config: GSAConfig):
        super().__init__()
        self.config = config

        # Slot parameters
        self.slots = nn.Parameter(torch.randn(config.num_slots, config.slot_size))

        # Projections
        self.q_proj = nn.Linear(config.hidden_size, config.slot_size)
        self.k_proj = nn.Linear(config.hidden_size, config.slot_size)
        self.v_proj = nn.Linear(config.hidden_size, config.slot_size)
        self.gate_proj = nn.Linear(config.hidden_size, config.num_slots)

        # Output projection
        self.o_proj = nn.Linear(config.slot_size, config.hidden_size)
        self.dropout = nn.Dropout(config.slot_dropout)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.size(0)

        # Project queries, keys, values
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Compute gating weights
        gates = torch.softmax(
            self.gate_proj(hidden_states) / self.config.gate_temperature,
            dim=-1
        )

        # Slot attention
        slot_weights = torch.einsum('bnd,md->bnm', q, self.slots)
        if attention_mask is not None:
            slot_weights = slot_weights.masked_fill(~attention_mask.unsqueeze(-1), float('-inf'))
        slot_weights = torch.softmax(slot_weights, dim=-1)

        # Update slots and compute output
        slot_values = torch.einsum('bnm,bnd->bmd', slot_weights, v)
        gated_values = torch.einsum('bnm,bmd->bnd', gates, slot_values)

        output = self.o_proj(self.dropout(gated_values))

        return output, gates