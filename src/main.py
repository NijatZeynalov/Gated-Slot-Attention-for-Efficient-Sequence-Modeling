import torch
from pathlib import Path
import argparse
from src.models import GSATransformer
from src.data import GSADataset, get_dataloader
from src.training import GSATrainer
from src.utils import get_logger
from src.config import load_config
from transformers import AutoTokenizer

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="GSA-Transformer Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train")
    parser.add_argument("--model_path", type=str, help="Path to pretrained model")
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_type"])
    model = GSATransformer(config["model"])

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path)["model_state_dict"])

    # Prepare data
    train_dataset = GSADataset(
        data_path=config["data"]["train_file"],
        tokenizer=tokenizer,
        max_length=config["model"]["max_seq_length"]
    )

    train_loader = get_dataloader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    val_dataset = GSADataset(
        data_path=config["data"]["validation_file"],
        tokenizer=tokenizer,
        max_length=config["model"]["max_seq_length"]
    )

    val_loader = get_dataloader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )

    # Initialize trainer
    trainer = GSATrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config["training"]
    )

    # Train or evaluate
    if args.mode == "train":
        logger.info("Starting training...")
        trainer.train()
    else:
        logger.info("Starting evaluation...")
        metrics = trainer.evaluate()
        logger.info(f"Evaluation metrics: {metrics}")


if __name__ == "__main__":
    main()