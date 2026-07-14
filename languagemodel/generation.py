import torch
from jaxtyping import Float, Int

from languagemodel.functional import softmax
from languagemodel.model import Transformer


def sample(
    logits: Float[torch.Tensor, "... vocab_size"],
    temperature: float = 1.0,
    top_p_threshold: float | None = None,
) -> Int[torch.Tensor, "1"]:
    logits = logits / max(temperature, 1e-5)
    probs = softmax(logits, dim=-1)

    if top_p_threshold is not None and top_p_threshold < 1.0:
        probs_sort, probs_idx = probs.sort(dim=-1, descending=True)

        probs_cumsum = probs_sort.cumsum(dim=-1)
        probs_sort.masked_fill_(probs_cumsum - probs_sort > top_p_threshold, 0)
        probs /= probs.sum(-1, keepdim=True)
        probs_sort.div_(torch.empty_like(probs_sort).exponential_(1))

        next_token_idx = probs_sort.argmax(dim=-1, keepdim=True)
        next_token = probs_idx[next_token_idx]

        return next_token

    probs.div_(torch.empty_like(probs).exponential_(1))
    next_token = probs.argmax(dim=-1, keepdim=True)

    return next_token


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: list[int],
    eos_id: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p_threshold: float | None = None,
    device: torch.device | None = None,
):
    model.eval()
    total_len = len(prompt_tokens) + max_new_tokens

    tokens = torch.full((total_len,), -1, device=device, dtype=torch.long)
    tokens[: len(prompt_tokens)] = torch.tensor(prompt_tokens)
    token_positions = torch.arange(total_len, device=device)

    for cur_pos in range(len(prompt_tokens), total_len, 1):
        prompt_mask = tokens != -1
        logits = model(tokens[prompt_mask], token_positions[prompt_mask])

        if temperature > 0.0:
            next_token = sample(logits[-1], temperature, top_p_threshold)
        else:
            next_token = logits[-1].argmax(dim=-1)

        if next_token.item() == eos_id:
            return tokens

        tokens[cur_pos] = next_token

    return tokens
