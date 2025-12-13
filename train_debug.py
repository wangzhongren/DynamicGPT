# train_debug.py
import os, torch, requests
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from model import DynaCatDiffusionLM
from dataset import TextDataset
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH = 64  # 减小 batch 便于调试
SEQ_LEN = 32  # 减小 seq_len 加速
LR = 3e-4
EPOCHS = 3
DATA_PATH = "data/tiny.txt"
BEST_CKPT = "model_best.pth"

# === 数据准备 ===
os.makedirs("data", exist_ok=True)
if not os.path.exists(DATA_PATH):
    try:
        with open("data/a.txt", "r", encoding='utf-8') as f:
            text = f.read()
        text = ''.join(c for c in text if 0 <= ord(c) < 128)
        with open(DATA_PATH, "w", encoding='utf-8') as f:
            f.write(text)
    except Exception as e:
        print(f"Failed to prepare data: {e}")
        exit(1)

dataset = TextDataset(DATA_PATH, seq_len=SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)
vocab_size = dataset.vocab_size
print(f"✅ Vocab size: {vocab_size}")

# === 模型与优化器 ===
model = DynaCatDiffusionLM(vocab_size, emb_dim=128, seq_len=SEQ_LEN).to(device)
opt = AdamW(model.parameters(), lr=LR)
criterion = CrossEntropyLoss()

# === 训练 ===
for epoch in range(EPOCHS):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"{'='*50}")

    for i, (x, y) in enumerate(tqdm(loader, desc="Step")):
        x, y = x.to(device), y.to(device)

        # === 调试打印 (仅前 2 个 step) ===
        if i < 2:
            print(f"\n--- Step {i} ---")
            # 打印 x, y 字符
            x_str = ''.join(dataset.itos[t] for t in x[0][:10])
            y_char = dataset.itos[y[0].item()]
            print(f"x[0][:10]: '{x_str}'")
            print(f"y[0]: '{y_char}'")
            print(f"x.shape: {x.shape}, y.shape: {y.shape}")

        opt.zero_grad()
        logits = model(x, use_refinement=False)  # 不用 refinement

        if i < 2:
            print(f"logits.shape: {logits.shape}")
            probs = F.softmax(logits[0], dim=-1)
            max_prob = probs.max().item()
            entropy = -(probs * probs.log()).sum().item()
            print(f"Max prob: {max_prob:.4f}, Entropy: {entropy:.2f}")

        loss = criterion(logits, y)
        loss.backward()

        # 梯度范数
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        if i < 2:
            print(f"Grad norm: {total_norm:.4f}")
            print(f"Loss: {loss.item():.4f}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if i >= 10:  # 只跑 10 步快速验证
            break

    # 计算小范围平均 loss
    print(f"\n→ Epoch {epoch+1} done (first 10 steps only).")

print("\n✅ Debug training completed.")