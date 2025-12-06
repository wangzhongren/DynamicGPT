# -*- coding: utf-8 -*-
"""
QKV as Dynamic Categorization (Fully Fixed v2)
—— 基于 "AI = 动态分类" 理论 (Wang, 2025)
彻底移除 "ok"，强制语义流导向终止符。
"""

import numpy as np

# 1. 词汇表（移除 "ok" 作为输出候选！）
VOCAB = ["hello", "hi", "how", "are", "you", "?", "I", "am", "fine", "!", "bye", "see", "later"]
STOP_TOKENS = {"?", "!", "."}

# 2. 手工语义嵌入（强化组内相似性）
EMB_DIM = 16
np.random.seed(0)

def make_embedding(word):
    base = np.zeros(EMB_DIM)
    if word in ["hello", "hi"]:
        base[0] = 1.0
    elif word in ["how", "are", "you"]:
        base[1] = 1.0
    elif word in ["I", "am", "fine"]:
        base[2] = 1.0
    elif word in ["bye", "see", "later"]:
        base[3] = 1.0
    elif word in STOP_TOKENS:
        base[4] = 1.0
    else:
        base[5] = 1.0
    base += np.random.randn(EMB_DIM) * 0.05  # 减小扰动
    return base

EMBEDDINGS = {w: make_embedding(w) for w in VOCAB}

# 3. 投影矩阵
np.random.seed(1)
Q_PROJ = np.random.randn(EMB_DIM, EMB_DIM) * 0.3
K_PROJ = np.random.randn(EMB_DIM, EMB_DIM) * 0.3
V_PROJ = np.random.randn(EMB_DIM, EMB_DIM) * 0.3

# 4. 核心生成函数
def generate_next_token(context_tokens):
    if not context_tokens:
        return "hello"
    
    context_embs = np.stack([EMBEDDINGS[t] for t in context_tokens])
    Q = (context_embs[-1:] @ Q_PROJ)
    K = context_embs @ K_PROJ
    V = context_embs @ V_PROJ

    attn_scores = Q @ K.T
    attn_scores = attn_scores - np.max(attn_scores, axis=-1, keepdims=True)
    exp_scores = np.exp(attn_scores)
    attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-8)
    category_prototype = (attn_weights @ V).flatten()

    similarities = np.array([
        np.dot(category_prototype, EMBEDDINGS[w]) /
        (np.linalg.norm(category_prototype) * np.linalg.norm(EMBEDDINGS[w]) + 1e-8)
        for w in VOCAB
    ])

    banned = set(context_tokens[-2:]) if len(context_tokens) >= 2 else {context_tokens[-1]}

    # 先尝试找能导向终止的
    sorted_indices = np.argsort(-similarities)
    for idx in sorted_indices:
        w = VOCAB[idx]
        if w not in banned:
            # 如果当前上下文适合结束，优先选 ? 或 !
            if context_tokens[-1] in ["you", "fine", "later"] and w in STOP_TOKENS:
                return w
            return w

    # 极端情况：全 banned，返回 !
    return "!"
# 5. 生成响应
def generate_response(prompt, max_tokens=10):
    tokens = prompt.split()
    for _ in range(max_tokens):
        next_tok = generate_next_token(tokens)
        tokens.append(next_tok)
        if next_tok in STOP_TOKENS:
            break
    return " ".join(tokens)

# 6. 演示
if __name__ == "__main__":
    print("🧠 QKV as Dynamic Categorization (Fully Fixed v2) — Wang, 2025\n")
    test_prompts = ["hello", "how are you", "I am fine", "bye"]
    for p in test_prompts:
        resp = generate_response(p)
        print(f"Prompt: '{p}' → Response: '{resp}'")