# train_stage1.py
import os, torch, requests
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from model import DynamicCategorizationLM
from dataset import TextDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH = 128
SEQ_LEN = 128
LR = 1e-4
EPOCHS = 100
DATA_PATH = "data/tiny.txt"
CKPT = "model_stage1.pth"

os.makedirs("data", exist_ok=True)
if not os.path.exists(DATA_PATH):
    try:
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        text = requests.get(url).text
        text = ''.join(c for c in text if 0 <= ord(c) < 128)
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"Failed to download: {e}")
        exit(1)

dataset = TextDataset(DATA_PATH, seq_len=SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)
vocab_size = dataset.vocab_size

model = DynamicCategorizationLM(vocab_size, emb_dim=256, seq_len=SEQ_LEN).to(device)
opt = AdamW(model.parameters(), lr=LR)
criterion = CrossEntropyLoss()

for epoch in range(EPOCHS):
    losses = []
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        pbar.set_postfix(loss=loss.item())

    avg_loss = sum(losses) / len(losses)
    print(f"Epoch {epoch+1}, avg loss: {avg_loss:.4f}")

torch.save({
    'state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'stoi': dataset.stoi,
    'itos': dataset.itos,
    'config': {'emb_dim': 256, 'seq_len': SEQ_LEN}
}, CKPT)
print(f"✅ Stage 1 model saved to {CKPT}")
