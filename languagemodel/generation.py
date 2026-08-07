import torch
from jaxtyping import Float, Int

from languagemodel.functional import softmax
from languagemodel.model import Transformer


def sample(
    logits: Float[torch.Tensor, "... vocab_size"],
    temperature: float = 1.0,
    top_p_threshold: float | None = None,
) -> Int[torch.Tensor, "1"]:
    """
    Amostra identificadores de tokens a partir de logits não normalizados

    Os logits são inicialmente divididos pela temperatura e convertidos em probabilidades com softmax. Quando "top_p_threshold" é fornecido, aplica-se nucleus sampling: somente os tokens mais prováveis cuja massa acumulada alcança o limiar são mantidos

    A amostragem categórica é realizada pelo método da corrida exponencial. Para cada token, uma variável exponencial é gerada e a função seleciona o índice que maximiza a razão entre sua probabilidade e o ruído amostrado

    Args:
        logits: Logits não normalizados sobre o vocabulário, com shape "(..., vocab_size)". temperature: Fator  utilizado para controlar a aleatoriedade da distribuição. Valores menores que 1 tornam a distribuição mais concentrada, valores maiores que 1 a tornam mais uniforme. Deve ser maior que zero.
        top_p_threshold: Limiar do nucleus sampling. Deve pertencer ao intervalo "(0, 1]". Quando "None' ou igual a 1, todos os tokens participam da amostragem.

    Returns:
        Tensor de índices inteiros com shape "(..., 1)", contendo um token amostrado para cada distribuição presente em "logits"
    """
    logits = logits / max(temperature, 1e-5)
    probs = softmax(logits, dim=-1)

    if top_p_threshold is not None and top_p_threshold < 1.0:
        probs_sort, probs_idx = probs.sort(dim=-1, descending=True)

        probs_cumsum = probs_sort.cumsum(dim=-1)
        probs_sort.masked_fill_(
            probs_cumsum - probs_sort > top_p_threshold,
            0,
        )
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
    """
    Gera tokens autorregressivamente a partir de um prompt

    A cada etapa, todos os tokens gerados até o momento são enviados ao modelo. Os logits da última posição são utilizados para selecionar o próximo token

    Quando "temperature" é maior que zero, a seleção utiliza amostragem categórica e pode aplicar nucleus sampling. Quando a temperatura é menor ou igual a zero, o token com maior logit é selecionado deterministicamente

    A geração termina quando o token EOS é produzido ou quando "max_new_tokens" tokens são gerados.

    A função processa apenas uma sequência por chamada

    Args:
        model: Transformer causal utilizado para calcular os logits. Seus parâmetros devem estar no mesmo dispositivo que os tokens
        prompt_tokens: Identificadores dos tokens que formam o prompt. A lista deve conter pelo menos um token
        eos_id: Identificador do token que encerra a geração
        max_new_tokens: Quantidade máxima de tokens que podem ser acrescentados ao prompt
        temperature: Fator que controla a aleatoriedade da geração. Valores positivos ativam amostragem, valores menores ou iguais a zero ativam seleção greedy
        top_p_threshold: Limiar opcional do nucleus sampling. É utilizado
     apenas quando ``temperature`` é maior que zero
        device: Device no qual os tokens e suas posições serão criados. Deve corresponder ao dispositivo do modelo. Quando "None", utiliza o dispositivo padrão do PyTorch

    Returns:
        Tensor unidimensional "torch.long" contendo os tokens do prompt seguidos dos tokens gerados. Seu comprimento pode ser menor que "len(prompt_tokens) + max_new_tokens" se o token EOS for produzido
    """
    model.eval()
    total_len = len(prompt_tokens) + max_new_tokens

    tokens = torch.full(
        (total_len,),
        -1,
        device=device,
        dtype=torch.long,
    )
    tokens[: len(prompt_tokens)] = torch.tensor(prompt_tokens)
    token_positions = torch.arange(total_len, device=device)

    for cur_pos in range(len(prompt_tokens), total_len, 1):
        prompt_mask = tokens != -1
        logits = model(
            tokens[prompt_mask],
            token_positions[prompt_mask],
        )

        if temperature > 0.0:
            next_token = sample(
                logits[-1],
                temperature,
                top_p_threshold,
            )
        else:
            next_token = logits[-1].argmax(dim=-1)

        if next_token.item() == eos_id:
            return tokens

        tokens[cur_pos] = next_token

    return tokens
