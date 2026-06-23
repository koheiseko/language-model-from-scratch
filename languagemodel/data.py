import torch
import numpy as np


@torch.no_grad()
def data_loading(
    dataset: np.ndarray | None = None,
    filename: str | None = None,
    batch_size: int = 32,
    context_length: int = 512,
    array_dtype: np.dtype | str | None = None,
) -> tuple[torch.Tensor]:
    """

    Args:
        dataset:
        filename:
        batch_size:
        context_length:
        array_dtype:
        tensor_dtype:
        device:
    """
    if dataset is None and filename is None:
        raise ValueError(
            "É preciso inserir um valor válido para o argumento 'dataset' ou 'filename'. Não forneça ambos."
        )

    if dataset is not None and filename is not None:
        raise ValueError(
            "Escolha utilizar apenas o argumento dataset ou apenas o filename."
        )

    if filename is not None and array_dtype is None:
        raise ValueError(
            "Ao utilizar o argumento 'filename', é obrigatório inserir um valor válido para o argumento 'array_dtype'."
        )

    if filename is not None:
        dataset = np.memmap(filename, dtype=array_dtype, mode="r")

    if dataset.size < context_length:
        raise ValueError(
            f"O tamanho do dataset ({dataset.size}) deve ser maior que o context length ({context_length})."
        )

    max_index = dataset.size - context_length
    indices = np.random.randint(0, max_index, (batch_size,))

    offsets = np.arange(context_length)
    indices_ids = indices[:, np.newaxis] + offsets

    ids = torch.tensor(
        dataset[indices_ids],
        dtype=torch.long,
    )
    targets = torch.tensor(
        dataset[indices_ids + 1],
        dtype=torch.long,
    )

    return ids, targets
