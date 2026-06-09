import torch
from jaxtyping import Float, Int


def softmax(input: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
    """
    Aplica a função softmax.

    Args:
        input:
        dim:
    """
    c = input.max(dim=dim, keepdim=True).values
    input_stable = input - c
    input_exp = input_stable.exp()

    return input_exp / input_exp.sum(dim=dim, keepdim=True)


def cross_entropy(logits: Float[torch.Tensor, "... seq_len vocab_size"], target: Int[torch.Tensor, "... seq_len"]):
    """
    Calcula a Cross Entropy Loss.

    Args:
        logits:
        target:
    """

    max_logits, _ = logits.max(dim=-1, keepdim=True)
    logits = logits - max_logits

    probs = -torch.gather(logits, index=target.unsqueeze(-1), dim=-1) + logits.exp().sum(dim=-1, keepdim=True).log()

    loss = probs.mean()

    return loss


def perplexity(losses: Float[torch.Tensor, "loss"]) -> torch.FloatType:
    """
    Calcula a Perplexidade.

    Args:
        losses:
    """

    return losses.mean().exp()
