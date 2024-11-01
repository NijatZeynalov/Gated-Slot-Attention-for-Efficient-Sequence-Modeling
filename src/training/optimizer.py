from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import get_scheduler
from typing import Optional
import torch.nn as nn


def create_optimizer(
        model: nn.Module,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
) -> Optimizer:
    # Separate parameters with and without weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_params = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    return AdamW(
        optimizer_params,
        lr=lr,
        betas=(beta1, beta2),
        eps=eps
    )


def create_scheduler(
        optimizer: Optimizer,
        num_training_steps: int,
        warmup_steps: Optional[int] = None,
        num_cycles: int = 1
) -> LRScheduler:
    if warmup_steps is None:
        warmup_steps = num_training_steps // 10

    return get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )