# train_stage2.py
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from model import DynaCatDiffusionLM
from dataset import TextDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH = 128
SEQ_LEN = 128
LR = 1e-5  # small LR
EPOCHS = 5
DATA_PATH = "data/tiny.txt"
CKPT_IN = "model_stage1.pth"
CKPT_OUT = "model_final.pth"

dataset = TextDataset(DATA_PATH, seq_len=SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

ckpt = torch.load(CKPT_IN, map_location=device)
vocab_size = ckpt['vocab_size']

# Load with cosine enabled
model = DynaCatDiffusionLM(vocab_size, emb_dim=256, seq_len=SEQ_LEN, use_cosine=True).to(device)
model.load_state_dict(ckpt['state_dict'], strict=False)  # out_proj missing, but ok

opt = AdamW(model.parameters(), lr=LR)
criterion = CrossEntropyLoss()

for epoch in range(EPOCHS):
    losses = []
    pbar = tqdm(loader, desc=f"Stage2 Epoch {epoch+1}")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x, use_refinement=False)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        pbar.set_postfix(loss=loss.item())

torch.save({
    'state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'config': {'emb_dim': 256, 'seq_len': SEQ_LEN, 'use_cosine': True}
}, CKPT_OUT)
print(f"✅ Final model saved to {CKPT_OUT}")