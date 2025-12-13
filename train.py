# train.py
import os, torch, requests
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from model import DynaCatDiffusionLM
from dataset import TextDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH = 128
SEQ_LEN = 128
LR = 3e-4
EPOCHS = 3
DATA_PATH = "data/tiny.txt"
BEST_CKPT = "model_best.pth"

os.makedirs("data", exist_ok=True)
if not os.path.exists(DATA_PATH):
    try:
        with open("data/a.txt","r+") as f:
            text = f.read();
        text = ''.join(c for c in text if 0 <= ord(c) < 128)
        with open(DATA_PATH, "w") as f:
            f.write(text)
    except Exception as e:
        print(f"Failed to download: {e}")
        exit(1)

dataset = TextDataset(DATA_PATH, seq_len=SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)
vocab_size = dataset.vocab_size

model = DynaCatDiffusionLM(vocab_size, emb_dim=256, seq_len=SEQ_LEN).to(device)
opt = AdamW(model.parameters(), lr=LR)
criterion = CrossEntropyLoss()

best_loss = float('inf')

for epoch in range(EPOCHS):
    if epoch == 8:
        for param_group in opt.param_groups:
            param_group['lr'] = 1e-4
        print("→ LR decayed to 1e-4")

    losses = []
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x, use_refinement=False)  # ← NO refinement in training
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        pbar.set_postfix(loss=loss.item())

    avg_loss = sum(losses) / len(losses)
    print(f"Epoch {epoch+1}, avg loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'state_dict': model.state_dict(),
            'vocab_size': vocab_size,
            'config': {'emb_dim': 256, 'seq_len': SEQ_LEN}
        }, BEST_CKPT)
        print(f"→ Saved best model (loss={avg_loss:.4f})")

torch.save(model.state_dict(), "model.pth")