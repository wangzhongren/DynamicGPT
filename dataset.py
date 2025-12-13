# dataset.py
import torch
import re

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, path, seq_len=128):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        # Tokenize: words + punctuation as tokens
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        unique_tokens = sorted(set(tokens))
        self.stoi = {t: i for i, t in enumerate(unique_tokens)}
        self.itos = {i: t for t, i in self.stoi.items()}
        self.vocab_size = len(unique_tokens)
        self.data = [self.stoi[t] for t in tokens]
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

    def encode(self, text):
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return [self.stoi.get(t, 0) for t in tokens]

    def decode(self, ids):
        return "".join([self.itos.get(i, "<unk>") for i in ids])