import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import logging
from pathlib import Path
from tqdm import tqdm
import wandb
from ..utils.logger import get_logger
from .optimizer import create_optimizer, create_scheduler

logger = get_logger(__name__)


class GSATrainer:
    def __init__(
            self,
            model: nn.Module,
            train_dataloader: DataLoader,
            val_dataloader: Optional[DataLoader] = None,
            config: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or {}

        # Setup training components
        self.optimizer = create_optimizer(
            model,
            lr=self.config.get('learning_rate', 2e-5)
        )

        num_training_steps = len(train_dataloader) * self.config.get('num_epochs', 3)
        self.scheduler = create_scheduler(
            self.optimizer,
            num_training_steps,
            warmup_steps=self.config.get('warmup_steps', None)
        )

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Setup mixed precision training
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.get('fp16', True))

        # Initialize tracking
        self.current_step = 0
        self.best_val_loss = float('inf')

        # Setup logging
        if self.config.get('use_wandb', False):
            wandb.init(project=self.config.get('project_name', 'gsa-transformer'))

    def train(self):

        num_epochs = self.config.get('num_epochs', 3)

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0

            with tqdm(self.train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
                for batch in pbar:
                    inputs, labels = batch["input_ids"].to(self.device), batch["labels"].to(self.device)

                    # Mixed precision training
                    with torch.cuda.amp.autocast(enabled=self.config.get('fp16', True)):
                        outputs = self.model(inputs)
                        loss = self.compute_loss(outputs, labels)

                    # Backpropagation
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    # Scheduler step
                    self.scheduler.step()

                    # Update progress
                    epoch_loss += loss.item()
                    pbar.set_postfix({"Loss": loss.item()})
                logger.info(f"Epoch {epoch + 1} completed. Average Loss: {epoch_loss / len(self.train_dataloader)}")
                """Main training loop."""
        num_epochs = self.config.get('num_epochs', 3)

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0

            with tqdm(self.train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
                for batch in pbar:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    # Forward pass with mixed precision
                    with torch.cuda.amp.autocast(enabled=self.config.get('fp16', True)):
                        outputs = self.model(**batch)
                        loss = outputs['loss']

                    # Backward pass
                    self.scaler.scale(loss).backward()

                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('max_grad_norm', 1.0)
                    )

                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # Update progress
                    epoch_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
                    self.current_step += 1

                    # Logging
                    if self.config.get('use_wandb', False):
                        wandb.log({
                            'train_loss': loss.item(),
                            'learning_rate': self.scheduler.get_last_lr()[0]
                        })

                    # Validation
                    if self.current_step % self.config.get('eval_steps', 500) == 0:
                        val_metrics = self.evaluate()
                        self.model.train()

                        # Save best model
                        if val_metrics['val_loss'] < self.best_val_loss:
                            self.best_val_loss = val_metrics['val_loss']
                            self.save_checkpoint('best_model.pt')

            # Save checkpoint
            if (epoch + 1) % self.config.get('save_epochs', 1) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')

    def evaluate(self):
        """Evaluation loop."""
        if not self.val_dataloader:
            return {}

        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs['loss'].item()

        val_loss = total_loss / len(self.val_dataloader)

        if self.config.get('use_wandb', False):
            wandb.log({'val_loss': val_loss})

        return {'val_loss': val_loss}

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        save_dir = Path(self.config.get('output_dir', 'outputs'))
        save_dir.mkdir(exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'step': self.current_step,
            'best_val_loss': self.best_val_loss
        }

        torch.save(checkpoint, save_dir / filename)
        logger.info(f'Saved checkpoint: {save_dir / filename}')