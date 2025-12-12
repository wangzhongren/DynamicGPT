# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicCategorizationLM(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.Wq = nn.Linear(emb_dim, hidden_dim)
        self.Wk = nn.Linear(emb_dim, hidden_dim)
        self.Wv = nn.Linear(emb_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.out_proj.weight = self.embedding.weight  # weight tying

    def forward(self, input_ids):
        B, L = input_ids.shape
        x = self.embedding(input_ids)  # (B, L, E)

        q = self.Wq(x[:, -1])          # (B, H) ← last token as query
        k = self.Wk(x)                 # (B, L, H)
        v = self.Wv(x)                 # (B, L, H)

        # Scaled dot-product attention → dynamic prototype
        attn = torch.einsum("bh,blh->bl", q, k) / (k.shape[-1] ** 0.5)
        attn = F.softmax(attn, dim=-1)
        prototype = torch.einsum("bl,blh->bh", attn, v)

        logits = self.out_proj(prototype)  # (B, vocab_size)
        return logits