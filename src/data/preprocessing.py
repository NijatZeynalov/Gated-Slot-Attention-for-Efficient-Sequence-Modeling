from transformers import PreTrainedTokenizer
from typing import List, Dict, Union, Optional
import json
from pathlib import Path


class Preprocessor:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def load_data(self, data_path: Union[str, Path]) -> List[str]:
        """Load and preprocess raw data."""
        data_path = Path(data_path)
        texts = []

        with open(data_path, 'r') as f:
            if data_path.suffix == '.json':
                data = json.load(f)
                texts = [item['text'] for item in data]
            else:
                texts = f.readlines()

        return [self.clean_text(text) for text in texts]

    def clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        return text.strip()


def tokenize_data(
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048
) -> Dict[str, List[List[int]]]:
    """Tokenize a batch of texts."""
    return tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )