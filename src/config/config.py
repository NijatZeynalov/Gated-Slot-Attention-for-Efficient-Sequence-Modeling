
import yaml
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import yaml
import os
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for the GSA Transformer model architecture."""
    model_type: str = "llama2"
    model_size: str = "7b"
    num_slots: int = 64
    max_seq_length: int = 2048
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    intermediate_size: int = 11008
    hidden_act: str = "silu"
    gsa_dropout: float = 0.1
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = field(default_factory=lambda: {"type": "linear", "factor": 1.0})

    # GSA-specific parameters
    slot_size: int = 256
    slot_dropout: float = 0.1
    gate_temperature: float = 1.0
    use_relative_positions: bool = True
    max_relative_position: int = 128

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    output_dir: str = "outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Optimizer settings
    optimizer_type: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100
    warmup_ratio: float = 0.01

    # Mixed precision training
    fp16: bool = True
    fp16_opt_level: str = "O2"

    # Distributed training
    local_rank: int = -1
    n_gpu: int = 1
    distributed: bool = False

    # Checkpointing
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 5

    # Logging
    logging_dir: str = "logs"
    logging_strategy: str = "steps"
    logging_steps: int = 100
    logging_first_step: bool = True

    # Evaluation
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    eval_accumulation_steps: Optional[int] = None

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class DataConfig:
    """Configuration for data processing and loading."""
    data_dir: str = "data"
    train_file: str = "train.json"
    validation_file: str = "validation.json"
    test_file: str = "test.json"

    # Preprocessing
    max_seq_length: int = 2048
    pad_to_max_length: bool = False
    truncation: bool = True
    preprocessing_num_workers: int = 4

    # Data loading
    shuffle: bool = True
    seed: int = 42
    overwrite_cache: bool = False

    # Tokenizer settings
    tokenizer_name: Optional[str] = None
    tokenizer_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for all components."""
    return {
        "model": ModelConfig().to_dict(),
        "training": TrainingConfig().to_dict(),
        "data": DataConfig().to_dict()
    }


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary containing configuration
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    # Merge with default config
    default_config = get_default_config()
    for section in default_config:
        if section not in config:
            config[section] = {}
        for key, value in default_config[section].items():
            config[section].setdefault(key, value)

    return config


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """
    Save configuration to a file.

    Args:
        config: Configuration dictionary
        save_path: Path to save configuration file
    """
    save_path = Path(save_path)
    os.makedirs(save_path.parent, exist_ok=True)

    if save_path.suffix == '.yaml' or save_path.suffix == '.yml':
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif save_path.suffix == '.json':
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config file format: {save_path.suffix}")


def create_config_from_args(args: Any) -> Dict[str, Any]:
    """
    Create configuration from command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Dictionary containing configuration
    """
    config = get_default_config()

    # Update config with command line arguments
    for section in config:
        section_config = config[section]
        for key in section_config:
            arg_value = getattr(args, f"{section}_{key}", None)
            if arg_value is not None:
                section_config[key] = arg_value

    return config

class Config:
    @classmethod
    def from_yaml(cls, file_path: str) -> 'Config':
        with open(file_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return cls(**config_dict)

    @classmethod
    def from_json(cls, file_path: str) -> 'Config':
        with open(file_path, 'r') as file:
            config_dict = json.load(file)
        return cls(**config_dict)

    def save_to_yaml(self, file_path: str) -> None:
        with open(file_path, 'w') as file:
            yaml.dump(self.to_dict(), file)
