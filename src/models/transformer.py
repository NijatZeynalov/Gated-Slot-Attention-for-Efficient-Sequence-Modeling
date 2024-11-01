import torch
import torch.nn as nn
from typing import Optional, Tuple
from .gsa import GSALayer, GSAConfig
from transformers import PreTrainedModel, PretrainedConfig


class GSATransformer(PreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.gsa_config = GSAConfig(
            hidden_size=config.hidden_size,
            num_slots=config.num_slots,
            slot_size=config.slot_size
        )

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # Layers
        self.layers = nn.ModuleList([
            GSALayer(self.gsa_config) for _ in range(config.num_hidden_layers)
        ])

        # Output head
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.gradient_checkpointing = False

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        # Get embeddings
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.embed_tokens(input_ids) + self.embed_positions(positions)

        # Apply GSA layers
        all_gates = []
        for layer in self.layers:
            x, gates = layer(x, attention_mask)
            all_gates.append(gates)

        # Output
        x = self.norm(x)
        logits = self.lm_head(x)

        # Loss calculation
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {"loss": loss, "logits": logits, "gates": all_gates}