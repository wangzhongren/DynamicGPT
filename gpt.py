# -*- coding: utf-8 -*-
"""
玩具 ChatGPT —— 基于 "AI = 动态分类" 理论 (Wang, 2025)
核心机制：每一步生成 = 在词汇表上做一次上下文相关的分类
"""


# 2. 基于上下文的“分类器”（模拟动态语义类别）
def classify_next_token(context_tokens):
    """
    输入：当前上下文 token 列表
    输出：从 VOCAB 中“分类”出最可能的下一个 token
    """
    if not context_tokens:
        return "hello"
    
    # 转为 tuple 便于匹配
    ctx = tuple(context_tokens)
    last = ctx[-1]
    last2 = ctx[-2:] if len(ctx) >= 2 else ()
    last3 = ctx[-3:] if len(ctx) >= 3 else ()

    # 规则：模拟“动态构建的语义类别”
    rules = {
        ("hello",): "hi",
        ("hi",): "how",
        ("how",): "are",
        ("how", "are"): "you",
        ("you",): "?",
        ("?",): "I",
        ("I",): "am",
        ("am",): "fine",
        ("fine",): "!",
        ("bye",): "see",
        ("see",): "you",
        ("you",): "later",  # 注意：和上面冲突，靠顺序优先

        ("what",): "is",
        ("what", "is"): "ai",
        ("ai",): "cool",
        ("cool",): "!",

        ("can",): "you",
        ("can", "you"): "do",
        ("do",): "math",
        ("math",): "2",
        ("2",): "+",
        ("+",): "4",
        ("4",): "=",
        ("=",): "6",
        ("6",): "!",

        ("why",): "is",
        ("why", "is"): "sky",
        ("sky",): "blue",
        ("blue",): "because",
        ("because",): "light",
        ("light",): "scatters",
        ("scatters",): "!",

        ("tell",): "me",
        ("tell", "me"): "about",
        ("about",): "cats",
        ("cats",): "are",
        ("are",): "nice",
        ("nice",): "!",

        # 默认 fallback
        "default": "ok"
    }

    # 优先匹配长上下文
    if last3 in rules:
        return rules[last3]
    if last2 in rules:
        return rules[last2]
    if (last,) in rules:
        return rules[(last,)]
    
    return rules["default"]

# 3. 自回归生成函数
def generate_response(prompt, history=[], max_tokens=15):
    """
    输入 prompt 和历史，返回模型生成的完整句子
    """
    # 合并历史和当前 prompt
    tokens = (history + prompt.split())
    generated = []

    for _ in range(max_tokens):
        next_tok = classify_next_token(tokens)
        if next_tok in ["!", "?", "."]:  # 简易停用
            generated.append(next_tok)
            break
        generated.append(next_tok)
        tokens.append(next_tok)  # 自回归：新 token 成为下一次输入

    return " ".join(generated)

# 4. 模拟多轮对话（非交互式，用于演示）
def demo_chat():
    history = []
    turns = [
        "hello",
        "what is ai",
        "can you do math",
        "why is sky blue",
        "tell me about cats",
        "bye"
    ]

    print("🤖 玩具 ChatGPT（基于‘动态分类’理论）\n")
    for user_input in turns:
        print(f"👤 User: {user_input}")
        response = generate_response(user_input, history=history)
        print(f"🤖 AI:   {response}\n")
        # 更新历史（可选：只保留最近 N 轮）
        history = (user_input.split() + response.split())[-20:]

if __name__ == "__main__":
    demo_chat()