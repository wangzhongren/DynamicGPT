# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DynamicCategorizationLM(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden_dim=256, seq_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(seq_len, emb_dim)
        self.Wq = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.Wk = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.Wv = nn.Linear(emb_dim, hidden_dim, bias=False)

        self.out_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.out_proj.weight = self.embedding.weight

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, input_ids):
        B, L = input_ids.shape
        x = self.embedding(input_ids)
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(pos_ids)

        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        attn_scores = torch.einsum("bqh,bkh->bqk", q, k) / (self.hidden_dim ** 0.5)
        mask = torch.tril(torch.ones(L, L, device=x.device))
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)

        out = torch.einsum("bqk,bkh->bqh", attn_weights, v)
        out = self.ffn(out)
        logits = self.out_proj(out)
        return logits  # ✅ 返回 [B, L, vocab_size]

    def build_prototype(self, context_ids):
        with torch.no_grad():
            L = context_ids.shape[0]
            if L > self.seq_len:
                context_ids = context_ids[-self.seq_len:]

            emb = self.embedding(context_ids.unsqueeze(0))  # [1, L, D]
            pos_ids = torch.arange(emb.shape[1], device=emb.device).unsqueeze(0)
            x = emb + self.pos_emb(pos_ids)  # ✅ Positional Encoding 没问题

            q = self.Wq(x[:, -1:])   # [1, 1, D]
            k = self.Wk(x)           # [1, L, D]
            v = self.Wv(x)           # [1, L, D]

            scores = torch.bmm(q, k.transpose(-2, -1)) / (self.hidden_dim ** 0.5)  # [1, 1, L]
            weights = F.softmax(scores, dim=-1)  # [1, 1, L]
            
            proto = torch.bmm(weights, v)        # [1, 1, D]  ← Attention 输出

            # --- 关键修复：加入 FFN 层 ---
            proto = self.ffn(proto)
            # ---------------------------

            return proto.squeeze(0).squeeze(0)

    def refine_prototype_stable(self, raw_proto, steps=20, noise_scale=0.3):
        device = raw_proto.device
        raw_np = raw_proto.cpu().numpy()
        x = np.random.randn(raw_np.shape[0])

        for i in range(steps):
            t = i / steps
            blend = 1.0 - np.exp(-5.0 * t)
            x = (1 - blend) * x + blend * raw_np
            if i < steps - 1:
                x += np.random.randn(x.shape[0]) * noise_scale * (1 - t)

        return torch.from_numpy(x).to(device).float()

    def generate_next_token(self, context_ids, itos, stoi, use_refinement=True):
        if isinstance(context_ids, list):
            context_ids = torch.tensor(context_ids, dtype=torch.long, device=next(self.parameters()).device)
        if len(context_ids) == 0:
            return "hello"

        proto = self.build_prototype(context_ids)

        if use_refinement:
            proto = self.refine_prototype_stable(proto)

        proto_norm = F.normalize(proto, dim=-1)
        emb_weight = F.normalize(self.embedding.weight, dim=-1)
        sims = torch.matmul(emb_weight, proto_norm)

        banned = set(context_ids[-2:].tolist()) if len(context_ids) >= 2 else set()
        for b in banned:
            sims[b] = -1e9

        # best_idx = sims.argmax().item()
        temperature = 0.000001  # 可调：0.7~1.0
        probs = F.softmax(sims / temperature, dim=-1)
        best_idx = torch.multinomial(probs, 1).item()
        return itos.get(best_idx, "<unk>")