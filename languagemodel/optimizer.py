import math

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        betas: list[float],
        weight_decay: float,
        eps: float,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
        }

        super().__init__(params, defaults)

    def step(self, closure=None):
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
                state = self.state[p]

                t = state.get("t", 1)
                m = state.get(
                    "m",
                    torch.zeros(
                        p.data.shape, device=p.data.device, dtype=p.data.dtype
                    ),
                )
                v = state.get(
                    "v",
                    torch.zeros(
                        p.data.shape, device=p.data.device, dtype=p.data.dtype
                    ),
                )

                lr_t = lr * (math.sqrt(1 - beta_2**t) / (1 - beta_1**t))

                p.data -= lr * weight_decay * p.data

                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * grad**2

                p.data -= lr_t * (m / (v.sqrt() + eps))

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss


def get_lr_cosine_schedule(
    t: int, lr_min: float, lr_max: float, t_w: int, t_c: int
) -> float:
    """
    Args:
        t:
        lr_min:
        lr_max:
        t_w:
        t_c:
    """
    if t < t_w:
        lr = (t / t_w) * lr_max
    elif t_w <= t <= t_c:
        decay_ratio = (t - t_w) / (t_c - t_w)
        lr = lr_min + 0.5 * (1.0 + math.cos(decay_ratio * math.pi)) * (
            lr_max - lr_min
        )
    else:
        lr = lr_min

    return lr


@torch.no_grad()
def gradient_clipping(parameters: list, l2_norm_max: float):
    if not isinstance(parameters, list):
        raise ValueError("Parameters deve ser uma lista.")

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

    if l2_norm > l2_norm_max:
        factor = l2_norm_max / (l2_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(factor)
