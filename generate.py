# generate.py
import torch
import argparse
from model import DynamicCategorizationLM

device = "cuda" if torch.cuda.is_available() else "cpu"

def main(prompt="bye", ckpt_path="model_stage1.pth", use_refinement=True, max_new_tokens=100):
    ckpt = torch.load(ckpt_path, map_location=device)
    
    vocab_size = ckpt['vocab_size']
    stoi = ckpt['stoi']
    itos = ckpt['itos']
    config = ckpt['config']

    model = DynamicCategorizationLM(
        vocab_size=vocab_size,
        emb_dim=config['emb_dim'],
        hidden_dim=config['emb_dim'],
        seq_len=config['seq_len']
    ).to(device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()

    # ✅ Use dataset-style tokenization
    import re
    tokens = re.findall(r"\w+|[^\w\s]", prompt, re.UNICODE)
    context_ids = [stoi.get(t, 0) for t in tokens]

    generated = []
    for _ in range(max_new_tokens):
        next_token = model.generate_next_token(
            context_ids=context_ids,
            itos=itos,
            stoi=stoi,
            use_refinement=use_refinement
        )
        generated.append(next_token)
        context_ids.append(stoi.get(next_token, 0))
        context_ids = context_ids[-128:]  # sliding window

        if next_token in {".", "!", "?", "\n"}:
            break

    full_output = prompt + " " + " ".join(generated)
    print(f'Prompt: "{prompt.strip()}"')
    print(f'Generated:\n{full_output}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default= "KING.\nWhat, ho!")
    parser.add_argument("--ckpt", type=str, default="model_stage1.pth")
    parser.add_argument("--no_refine", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=20)
    args = parser.parse_args()

    main(
        prompt=args.prompt,
        ckpt_path=args.ckpt,
        use_refinement=False,
        max_new_tokens=args.max_tokens
    )