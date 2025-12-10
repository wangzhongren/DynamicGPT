# -*- coding: utf-8 -*-
# goal_oriented_prototype_demo.py
# 2025 修正版：V 向量映射到“响应语义”，实现目标导向动态原型
# 完美复现合理语言行为

import numpy as np

# ============================ 1. 词汇表 ============================
VOCAB = ["hello", "hi", "how", "are", "you", "?", "I", "am", "fine", "!", "bye", "see", "later"]
STOP_TOKENS = {"?", "!", "."}

# ============================ 2. 手工语义嵌入（输入语义） ============================
EMB_DIM = 16
np.random.seed(0)

def make_input_embedding(word):
    base = np.zeros(EMB_DIM)
    if word in ["hello", "hi"]:           base[0] = 1.0   # 问候
    elif word in ["how", "are", "you"]:   base[1] = 1.0   # 提问
    elif word in ["I", "am", "fine"]:     base[2] = 1.0   # 自我陈述
    elif word in ["bye", "see", "later"]: base[3] = 1.0   # 告别
    elif word in STOP_TOKENS:             base[4] = 1.0   # 标点
    else:                                 base[5] = 1.0
    base += np.random.randn(EMB_DIM) * 0.05
    return base

INPUT_EMBEDDINGS = {w: make_input_embedding(w) for w in VOCAB}

# ============================ 3. 响应语义嵌入（用于 V 映射） ============================
# 这是关键修正：V 不再基于输入嵌入，而是基于“该词通常引发什么响应”
def get_response_embedding(word):
    """手工定义每个词的典型响应语义"""
    if word in ["how", "are", "you", "?"]:
        # 问题 → 回答（自我陈述）
        return INPUT_EMBEDDINGS["I"] * 0.4 + INPUT_EMBEDDINGS["am"] * 0.3 + INPUT_EMBEDDINGS["fine"] * 0.3
    elif word in ["hello", "hi"]:
        # 问候 → 问候（自反）
        return INPUT_EMBEDDINGS["hello"]
    elif word in ["I", "am"]:
        # 主语/系动词 → 表语（状态）
        return INPUT_EMBEDDINGS["fine"]
    elif word in ["fine", "!"]:
        # 陈述结束 → 告别或确认
        return INPUT_EMBEDDINGS["bye"]
    elif word in ["bye"]:
        # 告别 → 告别回应
        return (INPUT_EMBEDDINGS["see"] + INPUT_EMBEDDINGS["later"]) / 2.0
    elif word in ["see", "later"]:
        # 告别延续 → 结束
        return INPUT_EMBEDDINGS["!"]
    else:
        # 默认：自反
        return INPUT_EMBEDDINGS[word]

# 预计算所有词的响应向量（模拟训练好的 V_proj 效果）
RESPONSE_EMBEDDINGS = {w: get_response_embedding(w) for w in VOCAB}

# ============================ 4. Q/K 投影（模拟学习参数） ============================
np.random.seed(1)
Q_PROJ = np.random.randn(EMB_DIM, EMB_DIM) * 0.3
K_PROJ = np.random.randn(EMB_DIM, EMB_DIM) * 0.3

# 注意：V 不再通过矩阵投影，而是直接使用 RESPONSE_EMBEDDINGS
# 这等效于 V_proj 被“训练”为一个查表映射

# ============================ 5. 核心预测函数 ============================
def generate_next_token(context_tokens):
    if not context_tokens:
        return "hello"

    # Q 和 K 仍基于输入语义
    context_embs = np.stack([INPUT_EMBEDDINGS[t] for t in context_tokens])
    Q = context_embs[-1:] @ Q_PROJ
    K = context_embs @ K_PROJ

    # V 使用响应语义（关键！）
    V = np.stack([RESPONSE_EMBEDDINGS[t] for t in context_tokens])

    # 注意力（带缩放）
    scores = Q @ K.T / np.sqrt(EMB_DIM)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores) / (np.exp(scores).sum(axis=-1, keepdims=True) + 1e-8)

    # 动态原型 = 加权聚合的“响应语义”
    prototype = (weights @ V).flatten()

    # 与词汇表计算相似度（使用输入嵌入，因词汇表是输入空间）
    sims = []
    for w in VOCAB:
        dot = np.dot(prototype, INPUT_EMBEDDINGS[w])
        norm_p = np.linalg.norm(prototype)
        norm_w = np.linalg.norm(INPUT_EMBEDDINGS[w])
        sim = dot / (norm_p * norm_w + 1e-8)
        sims.append(sim)

    # 防重复
    banned = set(context_tokens[-2:]) if len(context_tokens) >= 2 else {context_tokens[-1]}

    # 选最相似且不重复的
    for idx in np.argsort(-np.array(sims)):
        w = VOCAB[idx]
        if w not in banned:
            return w

    return "!"

# ============================ 6. 测试 ============================
if __name__ == "__main__":
    print("✅ 修正版：目标导向动态原型（Wang, 2025 Revised）\n")
    tests = [
        ["hello"],
        ["hello", "how"],
        ["hello", "how", "are"],
        ["hello", "how", "are", "you"],
        ["how", "are", "you"],
        ["I", "am"],
        ["I", "am", "fine"],
        ["fine", "!"],
        ["bye"],
        ["hi"],
        ["see"],
    ]

    print(f"{'输入':<25} → 下一个词")
    print("-" * 40)
    for t in tests:
        next_word = generate_next_token(t)
        print(f"{' '.join(t):<25} → {next_word}")