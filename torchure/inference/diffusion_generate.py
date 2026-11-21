"""
sampling from a masked diffusion lm (objectives/masked_diffusion.py), with
llada's low-confidence remasking: start from the prompt followed by
`gen_len` [MASK]s, and at each of `steps` denoising steps predict every
masked position but commit only the most confident ones, so that
gen_len / steps tokens are revealed per step.

    uv run -m torchure.inference.diffusion_generate \
        --config configs/qwen3_diffusion_climbmix.json --step 1000 --prompt "The capital of"
"""

import argparse

import torch

from torchure.inference.common import load_config, load_model_and_tokenizer


@torch.no_grad()
def diffusion_generate(
    model,
    prompt_ids: torch.Tensor,  # (B, P)
    gen_len: int,
    steps: int,
    mask_id: int,
    temperature: float = 0.0,
) -> torch.Tensor:
    """(B, gen_len) generated ids."""
    B, P = prompt_ids.shape
    masks = torch.full((B, gen_len), mask_id, dtype=prompt_ids.dtype, device=prompt_ids.device)
    x = torch.cat([prompt_ids, masks], dim=1)

    for step in range(steps):
        masked = x == mask_id
        # every row has the same number of masks left: reveal an even share
        left = int(masked[0].sum())
        if left == 0:
            break
        k = -(-left // (steps - step))

        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            logits = model(x).float()
        # [MASK] is never a target, but an undertrained model can still pick it
        logits[..., mask_id] = float("-inf")
        if temperature > 0:
            # gumbel-max: a sample from softmax(logits / temperature)
            gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-20)))
            pred = (logits / temperature + gumbel).argmax(-1)
        else:
            pred = logits.argmax(-1)
        confidence = logits.softmax(-1).gather(-1, pred.unsqueeze(-1)).squeeze(-1)
        confidence = confidence.masked_fill(~masked, -1.0)

        reveal = confidence.topk(k, dim=-1).indices
        x.scatter_(1, reveal, pred.gather(1, reveal))

    return x[:, P:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--gen-len", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64, help="denoising steps (<= gen-len)")
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    assert config["objective"]["name"] == "masked_diffusion", "not a masked diffusion config"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(config, args.step, device)

    prompt = torch.tensor([tokenizer.encode(args.prompt).ids], device=device)
    out = diffusion_generate(
        model, prompt, args.gen_len, args.steps,
        config["objective"]["config"]["mask_token_id"], args.temperature,
    )
    print(args.prompt + tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
