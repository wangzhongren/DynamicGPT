# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicCategorizationLM(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden_dim=256, seq_len=64):
        super().__init__()
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(seq_len, emb_dim)
        
        # QKV projections
        self.Wq = nn.Linear(emb_dim, hidden_dim)
        self.Wk = nn.Linear(emb_dim, hidden_dim)
        self.Wv = nn.Linear(emb_dim, hidden_dim)
        
        # Output projection with weight tying
        self.out_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.out_proj.weight = self.embedding.weight  # weight tying
        
        # FFN for semantic abstraction (key for dynamic categorization)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.hidden_dim = hidden_dim

    def forward(self, input_ids):
        B, L = input_ids.shape
        x = self.embedding(input_ids)  # (B, L, emb_dim)

        # Add positional encoding (ensure L <= seq_len)
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0)  # (1, L)
        x = x + self.pos_emb(pos_ids)  # (B, L, emb_dim)

        # Compute Q, K, V (use emb_dim → hidden_dim)
        q = self.Wq(x)  # (B, L, hidden_dim)
        k = self.Wk(x)  # (B, L, hidden_dim)
        v = self.Wv(x)  # (B, L, hidden_dim)

        # Scaled dot-product attention with causal mask
        attn_scores = torch.einsum("bqh,bkh->bqk", q, k) / (self.hidden_dim ** 0.5)  # (B, L, L)
        mask = torch.tril(torch.ones(L, L, device=x.device))  # (L, L)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, L, L)

        out = torch.einsum("bqk,bkh->bqh", attn_weights, v)  # (B, L, hidden_dim)

        # Apply FFN: enables richer semantic prototype formation (critical!)
        out = self.ffn(out)  # (B, L, hidden_dim)

        # Project to vocab
        logits = self.out_proj(out)  # (B, L, vocab_size)

        # Predict next token after the last input token (for autoregressive training)
        return logits[:, -1, :]  # (B, vocab_size)