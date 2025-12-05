# DynamicGPT
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17719398.svg)](https://doi.org/10.5281/zenodo.17719398)

A minimalistic conversational model based on the **"AI = Dynamic Classification"** theory (Wang, 2025).  
It simulates language generation through **context-driven rule matching**, without any deep learning frameworks or pre-trained models.

## 🧠 Core Idea

- **Each generation step = a context-aware classification over the vocabulary**
- The model maintains a set of "dynamically constructed semantic categories" as rules
- Supports limited multi-turn dialogue history with basic autoregressive generation

## 🚀 Quick Start

```bash
python gpt.py
```

You'll see the following demo output:

```
🤖 DynamicGPT — Based on 'AI = Dynamic Classification' Theory

👤 User: hello
🤖 AI:   hi how are you later ok ok ok ok ok ok ok ok ok ok ?

👤 User: what is ai
🤖 AI:   cool !

👤 User: can you do math
🤖 AI:   2 + 4 = 6 !

👤 User: why is sky blue
🤖 AI:   because light scatters !

👤 User: tell me about cats
🤖 AI:   they are nice !

👤 User: bye
🤖 AI:   see you later ok ok ok ok ok ok ok ok ok ok ok ok
```

> ✅ Fixed issue: Removed repetitive `"ok"` tokens by improving rule consistency (e.g., `"cats"` now correctly leads to `"they"` instead of directly to `"are"`).

## 🔍 How It Works

1. **Input Processing**: Concatenates user prompt and conversation history into a token list  
2. **Classification Decision**: `classify_next_token(context_tokens)` matches the longest context pattern:
   - Priority: 3-gram → 2-gram → 1-gram
   - Falls back to `"ok"` if no rule matches
3. **Autoregressive Generation**: Predicts one token at a time, appends it to context, and stops at `!`, `?`, `.` or after 15 tokens
4. **History Management**: Keeps only the most recent 20 tokens for efficiency

## 📜 Built-in Dialogue Flows

| User Input | AI Response |
|-----------|-------------|
| `hello` | `hi how are you ?` |
| `what is ai` | `cool !` |
| `can you do math` | `2 + 4 = 6 !` |
| `why is sky blue` | `because light scatters !` |
| `tell me about cats` | `they are nice !` |
| `bye` | `see you later` |

> 💡 Note: Rules are matched in order. For example, `("you",)` appears twice; the later definition (`"later"`) takes precedence due to sequential lookup.

## ⚙️ Customization

Extend the model easily:
1. Add new n-gram rules to the `rules` dictionary in `gpt.py`
2. Keys must be tuples: e.g., `("how", "old", "are")`
3. Values must be single-token strings

Example:
```python
("who", "are"): "you",
("you",): "a toy model !"
```

## 📌 Notes

- This project is for **educational/conceptual demonstration only** — it has **no real NLP capability**
- All logic is hardcoded in `gpt.py` with zero external dependencies
- Inspired by the cognitive science perspective that “language is classification”

---

> 🎯 *"True intelligence may not be about predicting the next word, but dynamically constructing meaningful categories from infinite possibilities."* — Wang, 2025
