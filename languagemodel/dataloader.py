import numpy as np
import torch


@torch.no_grad()
def data_loading(
    dataset: np.ndarray | None = None,
    filename: str | None = None,
    batch_size: int = 32,
    context_length: int = 512,
    array_dtype: np.dtype | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Amostra um batch de sequências para treinamento autorregressivo

    Os tokens podem ser fornecidos diretamente por meio de um array NumPy ou carregados de um arquivo binário com "numpy.memmap". Exatamente uma dessas fontes deve ser informada

    Para cada elemento do batch, a função escolhe aleatoriamente uma posição inicial e extrai "context_length" tokens consecutivos. Os targets são formados pelos mesmos intervalos deslocados uma posição para a direita

    A amostragem é realizada com reposição, portanto duas linhas do batch podem corresponder à mesma posição inicial

    Args:
        dataset: Array unidimensional contendo a sequência completa de identificadores de tokens. Deve ser  fornecido somente quando "filename" for "None"
        filename: Caminho para um arquivo binário contendo uma sequência contígua de tokens. O arquivo é aberto em modo somente leitura usando "numpy.memmap". Deve ser fornecido somente quando "dataset" for "None"
        batch_size: Quantidade de sequências amostradas
        context_length: Quantidade de tokens em cada sequência de entrada e target
        array_dtype: Tipo numérico dos elementos armazenados no arquivo binário. É obrigatório quando "filename" é fornecido e não é utilizado quando "dataset" é passado diretamente.

    Returns:
        Tupla contendo:

        - "ids": tensor "torch.long" com shape "(batch_size, context_length)"
        - "targets": tensor "torch.long" com o mesmo shape, contendo os tokens de "ids" deslocados uma posição para a direita
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
