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
    out: str | os.PathLike,
) -> None:
    """
    Serializa o estado de um treinamento em um arquivo de checkpoint

    O checkpoint contém os parâmetros e buffers persistentes do modelo, o
    estado interno do otimizador, os argumentos necessários para reconstruir
    ambos, a loss associada ao checkpoint e o passo atual do treinamento

    O arquivo é salvo com "torch.save". Se já existir um arquivo no caminho
    informado, ele será sobrescrito. O diretório pai deve existir antes da
    chamada.

    Args:
        model: Modelo cujo "state_dict" será salvo
        optimizer: Otimizador cujo "state_dict" será salvo
        model_args: Argumentos necessários para reconstruir o modelo antes de carregar seus parâmetros
        optimizer_args: Argumentos necessários para reconstruir o otimizador antes de carregar seu estado
        loss: Valor da loss associado ao checkpoint
        step: Índice do passo de treinamento associado ao estado salvo
        out: Caminho do arquivo no qual o checkpoint será armazenado

    Returns:
        "None".
    """
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
    """
    Carrega um checkpoint e reconstrói o modelo e o otimizador

    O arquivo é desserializado com "torch.load". Em seguida, o modelo e o otimizador são instanciados usando os argumentos armazenados no checkpoint, e seus estados são restaurados por meio de "load_state_dict"

    Args:
        src: Caminho para o checkpoint ou objeto binário aberto para leitura
        model_module: Classe ou construtor do modelo. Deve aceitar os argumentos armazenados em "model_args"
        optimizer_module: Classe ou construtor do otimizador. Deve aceitar o argumento "params" e os valores armazenados em "optimizer_args"
        device: Device para o qual os tensores serializados serão remapeados durante o carregamento. Quando "None", preserva os dispositivos registrados no checkpoint

    Returns:
        Dicionário contendo:

        - "model": modelo reconstruído com o estado carregado
        - "optimizer": otimizador reconstruído com o estado carregado
        - "step': passo de treinamento registrado no checkpoint
    """

    checkpoint = torch.load(src, map_location=device)

    step = checkpoint["step"]
    loss = checkpoint["loss"]

    model_args = checkpoint["model_args"]
    optimizer_args = checkpoint["optimizer_args"]

    model_state_dict = checkpoint["model"]
    optimizer_state_dict = checkpoint["optimizer"]

    model = model_module(**model_args)
    model.load_state_dict(model_state_dict)

    param_dict = {
        pn: p for pn, p in model.named_parameters() if p.requires_grad
    }

    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

    optim_group = [
        {
            "params": decay_params,
            "weight_decay": optimizer_args["weight_decay"],
        },
        {
            "params": nodecay_params,
            "weight_decay": 0.0,
        },
    ]

    optimizer = optimizer_module(params=optim_group, **optimizer_args)
    optimizer.load_state_dict(optimizer_state_dict)

    return {
        "model": model,
        "optimizer": optimizer,
        "model_args": model_args,
        "optimizer_args": optimizer_args,
        "loss": loss,
        "step": step,
    }
