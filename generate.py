# generate.py
import torch, torch.nn.functional as F
from model import DynamicCategorizationLM

device = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "model_best.pth"
SEQ_LEN = 64

model = DynamicCategorizationLM(128, emb_dim=256, hidden_dim=256)
model.load_state_dict(torch.load(CKPT)['state_dict'])
model.to(device).eval()

def encode(text): return [min(ord(c), 127) for c in text]
def decode(ids): return ''.join(chr(i) for i in ids if 0 <= i < 128)

# ✅ 使用真实莎士比亚开头
prompt = "KING:\nIs this a dagger which I see before me,"
context = encode(prompt)
print(prompt)

with torch.no_grad():
    for _ in range(300):
        x = torch.tensor([context[-SEQ_LEN:]], device=device)
        logits = model(x)[0] / 0.8  # temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        context.append(next_token)

print(decode(context[len(encode(prompt)):]))