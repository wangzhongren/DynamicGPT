# generate.py
import torch
import torch.nn.functional as F
from model import DynamicCategorizationLM

device = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "model_best.pth"
SEQ_LEN = 128  # 必须与训练时一致

# === 1. 加载模型和元数据 ===
ckpt = torch.load(CKPT, map_location=device)
vocab_size = ckpt.get('vocab_size', 128)  # 优先用保存的 vocab_size

model = DynamicCategorizationLM(
    vocab_size=vocab_size,
    emb_dim=256,
    hidden_dim=256,
    seq_len=256
)
model.load_state_dict(ckpt['state_dict'])
model.to(device).eval()

# === 2. 编解码函数（确保只处理 0-127）===
def encode(text):
    return [min(ord(c), 127) for c in text]

def decode(ids):
    return ''.join(chr(i) for i in ids if 0 <= i < 128)

# === 3. 生成函数（带温度 + 重复惩罚）===
def generate(prompt, max_new_tokens=300, temperature=0.8, repetition_penalty=1.2):
    context = encode(prompt)
    print(prompt, end='', flush=True)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 取最后 SEQ_LEN 个 token
            input_ids = torch.tensor([context[-SEQ_LEN:]], device=device)
            logits = model(input_ids)[0]  # (vocab_size,)

            # 重复惩罚：降低最近 token 的 logit
            for token in set(context[-10:]):  # 惩罚最近10个
                if 0 <= token < vocab_size:
                    logits[token] /= repetition_penalty

            # Temperature + 采样
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            # 防止越界
            if next_token >= vocab_size:
                next_token = 0  # fallback to null or space

            context.append(next_token)
            print(chr(next_token) if 0 <= next_token < 128 else '?', end='', flush=True)

            # 可选：遇到换行+大写或特定符号提前停
            # if next_token == ord('\n'): break

    print()  # 换行
    return decode(context)

# === 4. 启动生成 ===
if __name__ == "__main__":
    prompt = "KING:\nIs this a dagger which I see before me,\n"
    generate(prompt, max_new_tokens=300, temperature=0.8, repetition_penalty=1.2)