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
        """
        Implementação do otimizador AdamW

        Args:
            params:
            lr:
            betas:
            weight_decay:
            eps:
        """
        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """ """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                t = state["t"]
                m = state["m"]
                v = state["v"]

                lr_t = lr * (math.sqrt(1 - beta_2**t) / (1 - beta_1**t))

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                m.mul_(beta_1).add_(grad, alpha=1 - beta_1)
                v.mul_(beta_2).addcmul_(grad, grad, value=1 - beta_2)

                p.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)

                state["t"] = t + 1

        return loss
