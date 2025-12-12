# train.py
import os, torch, requests
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from model import DynamicCategorizationLM
from dataset import TextDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 128
SEQ_LEN = 64
LR = 3e-4
EPOCHS = 10
DATA_PATH = "data/tiny.txt"
CKPT = "model.pth"

os.makedirs("data", exist_ok=True)
if not os.path.exists(DATA_PATH):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    mirror_url = "https://ghproxy.com/" + url
    # text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text
    with open("data/a.txt","r+") as f:
        text = f.read();
    text = ''.join(c for c in text if 0 <= ord(c) < 128)
    with open(DATA_PATH, "w") as f:
        f.write(text)

dataset = TextDataset(DATA_PATH, seq_len=SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

best_loss = float('inf')
best_path = "model_best.pth"

model = DynamicCategorizationLM(128, emb_dim=512, hidden_dim=256).to(device)
opt = AdamW(model.parameters(), lr=LR)
criterion = CrossEntropyLoss()

for epoch in range(EPOCHS):
    epoch_losses = []
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        
        # ✅ 关键：梯度裁剪（防震荡）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        opt.step()
        epoch_losses.append(loss.item())
        pbar.set_postfix(loss=loss.item())
    
    # 计算 epoch 平均 loss（更稳定）
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    print(f"Epoch {epoch+1}, avg loss: {avg_loss:.4f}")
    
    # ✅ 保存最佳模型
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'loss': avg_loss
        }, best_path)
        print(f"→ New best model saved (loss={avg_loss:.4f})")

torch.save({'state_dict': model.state_dict()}, CKPT)