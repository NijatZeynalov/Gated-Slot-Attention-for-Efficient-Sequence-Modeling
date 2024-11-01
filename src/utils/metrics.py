import torch
from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute model evaluation metrics."""
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=-1)

    pred_flat = predictions.cpu().numpy()
    labels_flat = labels.cpu().numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_flat,
        pred_flat,
        average='weighted'
    )

    return {
        'accuracy': accuracy_score(labels_flat, pred_flat),
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def calculate_perplexity(loss: torch.Tensor) -> float:
    """Calculate perplexity from loss."""
    return torch.exp(loss).item() if isinstance(loss, torch.Tensor) else np.exp(loss)