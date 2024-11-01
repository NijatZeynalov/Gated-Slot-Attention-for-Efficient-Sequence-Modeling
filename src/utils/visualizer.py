import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import numpy as np


def plot_attention_maps(
        attention_weights: torch.Tensor,
        tokens: List[str],
        layer_idx: Optional[int] = None,
        save_path: Optional[str] = None
):
    """Plot attention heatmaps."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights.cpu().numpy(),
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        annot=False
    )

    title = f"Attention Map - Layer {layer_idx}" if layer_idx is not None else "Attention Map"
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_training_metrics(
        metrics: Dict[str, List[float]],
        save_path: Optional[str] = None
):
    """Plot training metrics over time."""
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    for (metric_name, values), ax in zip(metrics.items(), axes):
        ax.plot(values)
        ax.set_title(f'{metric_name} over Time')
        ax.set_xlabel('Steps')
        ax.set_ylabel(metric_name)
        ax.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()