import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer

class GSADataset(Dataset):
    def __init__(self,
                 texts: List[str],
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 2048):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

def get_dataloader(
    dataset: GSADataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )