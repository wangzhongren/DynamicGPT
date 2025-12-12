# Dynamic Categorization Language Model

**A Unified View of AI Progress Through the Lens of Categorization**

This project demonstrates a novel perspective on artificial intelligence: **AI as dynamic categorization**. Rather than viewing language models as mere sequence predictors, we frame them as systems that continuously construct and refine semantic categories in context—a process that unifies perception, reasoning, and generation.

---

## 🧠 Core Idea

> **Every token prediction is a context-sensitive classification act.**

In this view:
- The vocabulary represents a fixed set of atomic symbols.
- At each generation step, the model *dynamically constructs a semantic category* (e.g., “greeting,” “math answer,” “emotional response”) based on context.
- The next token is selected by classifying which symbol best belongs to the current emergent category.

This reframing bridges symbolic AI (categories as concepts) and connectionist AI (neural representations), offering a unified lens for understanding AI progress.

---

## 📁 Project Structure

```
dynamic_categorization_lm/
├── gpt.py          # Toy ChatGPT: rule-based dynamic classifier (illustrates core idea)
├── model.py        # Neural implementation: attention + FFN as dynamic categorizer
├── train.py        # Training loop on Tiny Shakespeare
├── generate.py     # Text generation with sampling, repetition penalty, and temperature
└── README.md       # You are here!
```

---

## 🔧 Components Explained

### 1. `gpt.py` – The Conceptual Prototype
A minimal, rule-based "ChatGPT" that explicitly maps contexts to next tokens using hand-coded semantic rules (e.g., `("hello",) → "hi"`).  
✅ **Purpose**: Illustrate how *each generation step is a classification decision* under the dynamic categorization framework.

### 2. `model.py` – Neural Dynamic Categorizer
Implements a lightweight transformer-style LM where:
- **Attention** computes context-aware relevance (defining category boundaries).
- **FFN layers** enable *semantic abstraction*, forming richer prototypes—critical for dynamic category formation.
- **Weight tying** ensures embedding and output spaces align, reinforcing categorization consistency.

### 3. `train.py` – Training Pipeline
- Trains on [Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt).
- Uses gradient clipping, learning rate decay, and best-model checkpointing.
- Treats text as byte-level (0–127 ASCII), keeping vocabulary small (`vocab_size=128`).

### 4. `generate.py` – Controlled Generation
- Loads trained model and generates text autoregressively.
- Features:
  - **Temperature sampling** for diversity.
  - **Repetition penalty** to avoid loops.
  - Safe decoding (only printable ASCII).

---

## ▶️ Quick Start

```bash
# Install dependencies
pip install torch tqdm requests

# Train the model (downloads data automatically)
python dynamic_categorization_lm/train.py

# Generate text
python dynamic_categorization_lm/generate.py
```

Or run the conceptual demo:
```bash
python dynamic_categorization_lm/gpt.py
```

---

## 🌟 Why This Matters

This project operationalizes the thesis that **intelligence = adaptive categorization**. By showing both a symbolic toy and a neural implementation under the same framework, we argue that:

> Modern LMs don’t just “predict words”—they *continuously invent context-dependent categories* and assign tokens to them.

This perspective offers a path toward more interpretable, compositional, and human-aligned AI.

---

*Inspired by Wang (2025): "AI = Dynamic Categorization"*  
*Code by the Dynamic Categorization Research Group*