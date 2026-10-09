"""
interactive text-continuation repl for a checkpoint.

base LM with no chat template, so it continues text rather than answering.
no kv cache: every token recomputes the full prefix.

    uv run -m torchure.inference.chat \
        --config configs/qwen3_dense_climbmix_ddp.json --step 42000
"""
import argparse

import torch

from torchure.inference.common import load_config, load_model_and_tokenizer


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
    eos_id: int | None,
) -> str:
    ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
        next_logits = logits[0, -1, :].float()

        if temperature > 0:
            next_logits = next_logits / temperature
            if top_k > 0:
                kth_value = torch.topk(next_logits, top_k).values[-1]
                next_logits[next_logits < kth_value] = float("-inf")
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
        else:
            next_id = next_logits.argmax().view(1)

        input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)
        if eos_id is not None and next_id.item() == eos_id:
            break

    return tokenizer.decode(input_ids[0].tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(config, args.step, device)
    eos_id = tokenizer.token_to_id(config["data"]["eos_token"])

    print(f"loaded {config['run_name']} @ step {args.step}. base model, no chat "
          f"template -- type a prompt to continue, ctrl-d to quit.\n")
    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        completion = generate(
            model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_k, device, eos_id
        )
        print(completion)
        print()


if __name__ == "__main__":
    main()
