# dataset.py
import torch
from torch.utils.data import Dataset
from tokenizer import SimpleTokenizer

class TextDataset(Dataset):
    def __init__(self, file_path, seq_len=128):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        # 确保只含 ASCII
        text = ''.join(c for c in text if 0 <= ord(c) < 128)
        self.tok = SimpleTokenizer()
        self.ids = self.tok.encode(text)
        self.seq_len = seq_len
        self.vocab_size = self.tok.vocab_size  # 128

    def __len__(self):
        return max(0, len(self.ids) - self.seq_len)

    def __getitem__(self, idx):
        # 确保有足够上下文
        x = self.ids[idx: idx + self.seq_len]
        y = self.ids[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)