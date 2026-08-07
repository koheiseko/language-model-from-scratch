import math
from collections.abc import Iterable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        """
        Inicializa o otimizador Adam com weight decay desacoplado

        O otimizador mantém, para cada parâmetro, médias móveis exponenciais do
        gradiente e do quadrado do gradiente. Essas estimativas são utilizadas
        para adaptar individualmente o tamanho das atualizações

        Diferentemente da regularização L2 aplicada ao gradiente, o AdamW aplica
        o weight decay diretamente aos parâmetros, antes da atualização
        adaptativa

        Args:
            params: Iterable de parâmetros a serem otimizados ou de dicionários definindo grupos de parâmetros com configurações próprias
            lr: Taxa de aprendizado base. Deve ser maior ou igual a zero
            betas: Coeficientes utilizados nas médias móveis do gradiente e do gradiente ao quadrado. Ambos devem pertencer ao intervalo "[0, 1)"
            eps: Constante adicionada ao denominador para melhorar a estabilidade numérica. Deve ser maior ou igual a zero
            weight_decay: Coeficiente do decaimento desacoplado aplicado aos parâmetros. Normalmente deve ser maior ou igual a zero

        Attributes:
            param_groups: Grupos de parâmetros administrados pelo otimizador Cada grupo pode possuir valores próprios para "lr", "betas", "eps" e "weight_decay"
            state: Estado mantido individualmente para cada parâmetro. Contém o número do passo e as médias móveis de primeira e segunda ordem
            defaults: Valores padrão usados na criação dos grupos de parâmetros
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Executa uma etapa de atualização dos parâmetros

        Para cada parâmetro que possui gradiente, o método atualiza as médias
        móveis de primeira e segunda ordem, aplica o weight decay desacoplado e
        realiza a atualização adaptativa com correção de viés

        Os estados "m" e "v" são armazenados em "torch.float32" para melhorar a estabilidade numérica, inclusive quando os parâmetros utilizam "float16" ou "bfloat16".

        Args:
            closure: Função opcional que recalcula o modelo e retorna a loss.

        Returns:
            O valor retornado por "closure", quando ela é fornecida, caso
            contrário, retorna "None".
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")

                state = self.state[p]

                t = state.get("t", 1)

                lr_t = lr * (math.sqrt(1 - beta_2**t) / (1 - beta_1**t))

                p.data -= lr * weight_decay * p.data

                prev_m_t = state.get(
                    "m",
                    torch.zeros(
                        p.data.shape,
                        device=p.data.device,
                        dtype=torch.float32,
                    ),
                )
                prev_v_t = state.get(
                    "v",
                    torch.zeros(
                        p.data.shape,
                        device=p.data.device,
                        dtype=torch.float32,
                    ),
                )

                m_t = beta_1 * prev_m_t + (1 - beta_1) * grad
                v_t = beta_2 * prev_v_t + (1 - beta_2) * grad**2

                p.data -= lr_t * m_t / (v_t.sqrt() + eps)

                state["t"] = t + 1
                state["m"] = m_t
                state["v"] = v_t

        return loss


def get_lr_cosine_schedule(
    it: int,
    min_learning_rate: float,
    max_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Dado os parâmetros do cosine learning decay schedule (com linear warmup) e o número da iteração, retorna o learning rate na iteração especificada, de acordo com o schedule definido.

    Args:
        it (int): Número da iteração que se deseja obter o learning rate.
        min_learning_rate (float): alpha_max, o maior valor para o learning rate para o cosine learning schedule (com warmup).
        max_learning_rate (float): alpha_min, o menor valor para o learning rate para o cosine learning schedule (com warmup).
        warmup_iters (int): T_w, número de iterações da etapa de warmup.
        cosine_cycle_iters (int): T_c, número de iteações da etapa de cosine annealing.

    Returns:
        Learning rate na iteração especificada, de acordo com o schedule definido.
    """
    if it < warmup_iters:
        lr = (it / warmup_iters) * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        lr = min_learning_rate + 0.5 * (
            1.0 + math.cos(decay_ratio * math.pi)
        ) * (max_learning_rate - min_learning_rate)
    else:
        lr = min_learning_rate

    return lr


@torch.no_grad()
def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    """
    Dado um conjunto de parâmetros, limita seus gradientes combinados de modo que l2 norm seja, no máximo, "max_l2_norm".

    Args:
        parameters (Iterable[torch.nn.Parameter]): Conjunto de parâmetros treináveis.
        max_l2_norm (float): Valor positivo máximo para a l2 norm.

    Os gradientes são modificados de forma in-place.
    """
    eps = 1e-6

    grad = torch.cat(
        [
            p.grad.flatten()
            if p.grad is not None
            else torch.zeros_like(p).flatten()
            for p in parameters
        ]
    )

    l2_norm = (grad**2).sum().sqrt()

    clip_coef = min(1, max_l2_norm / (l2_norm + eps))

    for p in parameters:
        if p.grad is not None:
            p.grad *= clip_coef
