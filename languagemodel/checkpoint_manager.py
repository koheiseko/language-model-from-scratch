import os
import typing

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_args: dict,
    optimizer_args: dict,
    loss: float,
    step: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()

    checkpoint = {
        "model": model_state_dict,
        "optimizer": optimizer_state_dict,
        "model_args": model_args,
        "optimizer_args": optimizer_args,
        "loss": loss,
        "step": step,
    }

    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model_module: torch.nn.Module,
    optimizer_module: torch.optim.Optimizer,
    device: torch.device | None = None,
) -> dict:
    checkpoint = torch.load(src, map_location=device)

    step = checkpoint["step"]

    model_args = checkpoint["model_args"]
    optimizer_args = checkpoint["optimizer_args"]

    model_state_dict = checkpoint["model"]
    optimizer_state_dict = checkpoint["optimizer"]

    model = model_module(**model_args)
    model.load_state_dict(model_state_dict)

    optimizer = optimizer_module(params=model.parameters(), **optimizer_args)
    optimizer.load_state_dict(optimizer_state_dict)

    return {"model": model, "optimizer": optimizer, "step": step}
