import torch
from jaxtyping import Float, Int


def softmax(
    input: Float[torch.Tensor, "..."], dim: int
) -> Float[torch.Tensor, "..."]:
    """
    Dado um tensor como input, retorna como output um tensor com a função softmax aplicado nos valores de acordo com a dimensão.

    Args:
        input (Float[torch.Tensor, "..."]): Tensor para a aplicação da função softmax. O shape é arbitrário.
        dim (int): Dimensão que deve ser aplicado a função softmax.
    Returns:
        Float[torch.Tensor, "..."]: Tensor com o mesmo shape do "input" normalizado através da função softmax de acordo com o valor de "dim".
    """
    c = input.max(dim=dim, keepdim=True).values
    input_stable = input - c
    input_exp = input_stable.exp()

    return input_exp / input_exp.sum(dim=dim, keepdim=True)


def cross_entropy(
    inputs: Float[torch.Tensor, "... sequence_length vocab_size"],
    targets: Int[torch.Tensor, "... sequence_length"],
) -> Float[torch.Tensor, ""]:
    """
    Dado um tensor como inputs e outro como targets, computa a média da cross entropy loss dos exemplos.

    Args:
        inputs (inputs: Float[torch.Tensor, "... sequence_length vocab_size"]): inputs[i][j] é o logit não normalizado da classe jth para o exemplo ith.
        targets (Int[torch.Tensor, "... sequence_length"]): Tensor contendo o index da classe correta para cada exemplo.
    Returns:
        Float[torch.Tensor, ""]: Média da cross entropy loss dos exemplos.
    """

    max_inputs, _ = inputs.max(dim=-1, keepdim=True)
    inputs = inputs - max_inputs

    log_sum_exp = (
        inputs.exp().sum(dim=-1, keepdim=True).log()
    )  # (... sequence_length)

    target_logit = torch.gather(inputs, index=targets.unsqueeze(-1), dim=-1)

    losses = log_sum_exp - target_logit  # (... sequence_length)

    return losses.mean()
