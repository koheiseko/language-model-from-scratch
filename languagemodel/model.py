import math
import warnings

import einops
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int

from languagemodel.functional import softmax


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Inicializa o módulo Linear sem termo de bias, seguindo a prática comum em LLMs

        Os pesos são inicializados a partir de uma distribuição normal truncada, com desvio padrão definido pela Glorot inicialization

        Args:
            in_features: Número de features do input
            out_features: Número de features do output
            device: Device aonde a tensor de pesos será inicializada Quando "None", utiliza o device padrão do PyTorch
            dtype: Type da tensor de pesos. Quando "None", utiliza o dtype padrão do PyTorch

        Atributos:
            in_features: Número de features do input
            out_features: Número de features do output
            weights: Parâmetro treinável com shape "(out_features, in_features)"
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Glorot inicialization
        var = 2 / (in_features + out_features)
        std = math.sqrt(var)
        tensor = torch.empty(
            out_features,
            in_features,
            device=device,
            dtype=dtype,
        )
        nn.init.trunc_normal_(
            tensor,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

        self.weights = nn.Parameter(tensor, requires_grad=True)

    def forward(
        self,
        x: Float[torch.Tensor, "... in_features"],
    ) -> Float[torch.Tensor, "... out_features"]:
        """
        Aplica a transformação linear sobre a última dimensão do input

        Args:
            x: Tensor de input

        Returns:
            Tensor de output, obtido através da transformação linear aplicado no input
        """

        return einops.einsum(
            x,
            self.weights,
            "... in_features, out_features in_features-> ... out_features",
        )

    def extra_repr(self):
        return f"in_features={self.weights.shape[1]}, out_features={self.weights.shape[0]}"


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Inicializa o Módulo de Embedding

        Os embeddings são inicializados com uma distribuição normal truncada, com média zero e desvio padrão 1, limitada ao intervalo [-3, 3]

        Args:
            num_embeddings: Número de embeddings de acordo com o tamanho do vocabulário
            embedding_dim: Tamanho da dimensão de cada embedding
            device: Device aonde o tensor de embeddings será inicializada. Quando "None", utiliza o device padrão do PyTorch
            dtype: Type do tensor de embeddings. Quando "None", utiliza o dtype padrão do PyTorch

        Atributos:
            weights: Matriz de embeddings treinável com shape (num_embeddings, embedding_dim)
        """
        super().__init__()
        tensor = torch.empty(
            num_embeddings,
            embedding_dim,
            device=device,
            dtype=dtype,
        )
        nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=-3, b=3)

        self.weights = nn.Parameter(tensor, requires_grad=True)

    def forward(
        self,
        token_ids: Int[torch.LongTensor, "..."],
    ) -> Float[torch.Tensor, "... embedding_dim"]:
        """
        Recupera os embeddings associados aos ids fornecidos

        Args:
            token_ids: Tensor de ids com o type Int e com shape (...)

        Returns:
            Tensor contendo os embeddings selecionados, com shape (..., embedding_dim)
        """

        return self.weights[token_ids, :]

    def extra_repr(self):
        return f"vocab_size={self.weights.shape[0]}, embedding_dim={self.weights.shape[1]}"


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Inicializa o Root Mean Squared Layer Normalization (RMSNorm)

        Args:
            dim: Tamanho da última dimensão do input a ser normalizada
            eps: Pequena constante adicionada à média dos quadrados para evitar
            divisão por zero e melhorar a estabilidade numérica
            device: Device aonde o tensor de pesos será inicializada. Quando "None", utiliza o device padrão do PyTorch
            dtype: Type do tensor de pesos. Quando "None", utiliza o dtype padrão do PyTorch

        Atributos:
            weights: Pesos treináveis, inicializados com valores iguais a 1 e com shape (dim,)
            eps: Constante utilizada para estabilidade numérica.
        """
        super().__init__()
        self.weights = nn.Parameter(
            torch.ones(dim, device=device, dtype=dtype),
            requires_grad=True,
        )
        self.eps = eps

    def forward(
        self, x: Float[torch.Tensor, "... dim"]
    ) -> Float[torch.Tensor, "... dim"]:
        """
        Para cada vetor da entrada, o módulo divide seus elementos pela raiz da
        média dos seus quadrados e aplica, em seguida, um fator de escala
        treinável

        O cálculo da normalização é realizado em "torch.float32" para melhorar
        a estabilidade numérica. Antes de ser retornado, o resultado é convertido
        novamente para o dtype original da entrada

        Args:
            x: Tensor de input

        Returns:
            Tensor normalizado com o mesmo shape e dtype da entrada.
        """
        in_dtype = x.dtype

        x = x.to(torch.float32)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        result = (x / rms) * self.weights

        return result.to(in_dtype)

    def extra_repr(self):
        return f"hidden_size={self.weights.shape[0]}, eps={self.eps}"


class SwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Implementa uma camada feed-forward baseada em SwiGLU

        O módulo projeta a entrada em dois caminhos. No primeiro, aplica a ativação SiLU. O resultado é multiplicado elemento a elemento pela segunda projeção. Por fim, uma terceira transformação linear projeta o resultado de volta à dimensão original:

        Args:
            theta:
            dim: Dimensão dos tensors de entrada e saída
            hidden_dim: Dimensão intermediária
            device: Device aonde dos tensors das projeções serão inicializada. Quando "None", utiliza o device padrão do PyTorch
            dtype: Type dos tensors das projeções. Quando "None", utiliza o dtype padrão do PyTorch

        Atributos:
            w1: Projeção linear de dim -> hidden_dim cuja saída recebe a ativação SiLU
            w2: Projeção linear de hidden_dim -> dim
            w3: Segunda projeção linear de dim -> para hidden_dim,
            utilizada no mecanismo de gating
        """
        super().__init__()

        self.w1_weight = Linear(
            dim,
            hidden_dim,
            device=device,
            dtype=dtype,
        )
        self.w2_weight = Linear(
            hidden_dim,
            dim,
            device=device,
            dtype=dtype,
        )
        self.w3_weight = Linear(
            dim,
            hidden_dim,
            device=device,
            dtype=dtype,
        )

    def forward(
        self, x: Float[torch.Tensor, "... dim"]
    ) -> Float[torch.Tensor, "... dim"]:
        """
        Aplica a transformação feed-forward SwiGLU

        Args:
            x: Tensor de input

        Returns:
            Tensor transformado com o mesmo shape do tensor de input
        """
        x1 = self.w1_weight(x)
        values = x1 * torch.sigmoid(x1)

        gates = self.w3_weight(x)

        return self.w2_weight(values * gates)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        dim: int,
        context_length: int,
        device: torch.device | None = None,
    ):
        """
        Inicializa o Rotary Position Embedding (RoPE)

        RoPE incorpora informação posicional rotacionando pares consecutivos de componentes dos vetores de entrada. O ângulo usado em cada rotação depende tanto da posição do token quanto da frequência associada ao par de componentes

        Os valores de seno e cosseno são pré-calculados para todas as posições até "context_length" e armazenados em um buffer não treinável

        Args:
            theta: Base utilizada para definir as frequências das rotações
            dim: Dimensão dos vetores sobre os quais RoPE será aplicado. Deve ser um número par, pois os componentes são processados em pares
            max_seq_len: Maior quantidade de posições pré-calculadas no cache
            device: Device no qual o cache de frequências será criado. Quando "None", utiliza o dispositivo padrão do PyTorch

        Atributos:
            _freq_cis_cache: Buffer não persistente contendo os valores de cosseno
            e seno pré-calculados, com shape (2, context_length, dim // 2). O índice 0 contém os cossenos e o índice 1 contém os senos

        """
        super().__init__()

        self.register_buffer(
            "_freq_cis_cache",
            RotaryPositionalEmbedding._init_cache(
                context_length, dim, theta, device
            ),
            persistent=False,
        )

        self._freq_cis_cache: Float[
            torch.Tensor,
            "2 max_seq_length half_dim",
        ]

    @staticmethod
    def _init_cache(
        context_length: int,
        dim: int,
        theta: float,
        device: torch.device | None = None,
    ) -> Float[torch.Tensor, " 2 context_length half_dim"]:
        """
        Pré-calcula os senos e cossenos utilizados nas rotações

        Para cada posição e cada par de componentes, calcula uma frequência angular e armazena seu cosseno e seno. O cache é criado em "torch.float32" para preservar estabilidade numérica
        """
        if dim % 2 != 0:
            raise ValueError(
                f'O valor de "dim" deve ser par, mas recebeu {dim}.'
            )

        pairs_counts = (
            torch.arange(
                0,
                dim,
                2,
                dtype=torch.float32,
                device=device,
            )
            / dim
        )
        t = torch.arange(
            context_length,
            dtype=torch.float32,
            device=device,
        )
        freqs = theta**-pairs_counts

        freqs = einops.einsum(
            t,
            freqs,
            "t, f -> t f",
        )

        cos, sin = freqs.cos(), freqs.sin()

        return torch.stack((cos, sin))

    def forward(
        self,
        x: Float[
            torch.Tensor,
            "... sequence_length dim",
        ],
        token_positions: Int[
            torch.Tensor,
            "... sequence_length",
        ]
        | None = None,
    ) -> Float[
        torch.Tensor,
        "... sequence_length dim",
    ]:
        """
        Rotaciona os pares de componentes de acordo com suas posições.

        Quando "token_positions" não é fornecido, são usadas sequencialmente as posições de zero até "sequence_length - 1". Quando fornecido, o tensor permite selecionar posições específicas do cache

        Args:
            x: Tensor sobre o qual RoPE será aplicado
            token_positions: Índices das posições de cada token no intervalo

        Returns:
            Tensor rotacionado com o mesmo shape e dtype de "x".
        """
        if x.size(-1) % 2 != 0:
            raise ValueError(
                f"A última dimensão de `x` deve ser par, mas recebeu {x.size(-1)}."
            )

        x1, x2 = x[..., 0::2], x[..., 1::2]

        if token_positions is not None:
            cos, sin = self._freq_cis_cache[
                :,
                token_positions,
                :,
            ].unbind(0)

        else:
            seq_len = x.size(-2)

            if seq_len > self._freq_cis_cache.size(1):
                raise ValueError(
                    f"Comprimento da sequência ({seq_len}) excede "
                    f"max_seq_len ({self._freq_cis_cache.size(1)})."
                )

            cos, sin = self._freq_cis_cache[
                :,
                :seq_len,
                :,
            ].unbind(0)

        cos = cos.to(dtype=x.dtype, device=x.device)
        sin = sin.to(dtype=x.dtype, device=x.device)

        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        result = torch.stack((x1_rot, x2_rot), dim=-1).flatten(-2)

        return result


def scaled_dot_product_attention(
    query: Float[torch.Tensor, "... queries d_k"],
    key: Float[torch.Tensor, "  ... keys    d_k"],
    value: Float[torch.Tensor, "... keys    d_v"],
    is_causal: bool,
    attn_mask: Bool[torch.Tensor, " ... queries keys"]
    | Float[torch.Tensor, "... queries keys"]
    | None = None,
) -> Float[torch.Tensor, "... queries d_v"]:
    """
    Calcula o Scaled Dot Product Attention (SDPA).

    Args:
        query: Tensor de queries
        key: Tensor de keys
        value: Tensor de values
        is_causal:  Se "True", impede que uma posição consulte posições futuras. Não pode ser utilizado junto com "attn_mask"
        attn_mask: Máscara opcional com shape compatível com (..., queries, keys). Quando booleana, valores "True" indicam posições permitidas e valores 'False" indicam posições que devem se0r ignoradas. Quando de ponto flutuante, seus valores são adicionados diretamente aos scores de atenção

    Returns:
        Tensor resultado da aplicação dos pesos de atenção no tensor de values
    """
    queries, keys = query.size(-2), key.size(-2)
    d_k = key.size(-1)
    scale_factor = 1 / math.sqrt(d_k)

    attn_scores = (
        einops.einsum(
            query,
            key,
            "... queries d_k, ... keys d_k -> ... queries keys",
        )
        * scale_factor
    )

    if is_causal:
        if attn_mask is not None:
            raise ValueError(
                "attn_mask não pode ser passado junto com is_causal=True"
            )

        causal_mask = torch.ones(
            queries,
            keys,
            dtype=torch.bool,
            device=query.device,
        ).triu(diagonal=1)
        attn_scores.masked_fill_(causal_mask, float("-inf"))

    if attn_mask is not None:
        if attn_mask.device != query.device:
            attn_mask = attn_mask.to(query.device)

        if attn_mask.dtype == torch.bool:
            attn_scores.masked_fill_(
                attn_mask.logical_not(),
                float("-inf"),
            )
        else:
            attn_scores = attn_scores + attn_mask.to(dtype=attn_scores.dtype)

    attn_weights = softmax(attn_scores, dim=-1)

    return einops.einsum(
        attn_weights,
        value,
        "... queries keys, ... keys d_v -> ... queries d_v",
    )


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        positional_encoder: RotaryPositionalEmbedding | None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Inicializa o mecanismo de MHA

        A entrada é projetada independentemente em queries, keys e values. Cada projeção é dividida em "num_heads" cabeças, permitindo que diferentes subespaços da representação sejam processados em paralelo

        Está implementação do MHA utiliza RoPE como codificar posicional, aonde ele é aplicado nas projeções de queries e keys

        Args:
            d_model: Dimensão dos vetores de entrada e saída. Deve ser divisível por "num_heads"
            num_heads: Quantidade de cabeças de atenção executadas em paralelo
            positional_encoder: Codificador posicional aplicado às queries e keys. Pode ser "None" para desativar a codificação posicional
            device: Device no qual os parâmetros das projeções serão criados. Quando "None", utiliza o dispositivo padrão do PyTorch
            dtype: Type dos parâmetros das projeções. Quando "None", utiliza o dtype padrão do PyTorch.

        Attributes:
            head_dim: Dimensão de cada cabeça, calculada como "d_model // num_heads"
            num_heads: Quantidade de cabeças de atenção
            positional_encoder: Codificador posicional aplicado às queries e keys.
            q_proj: Projeção linear que produz as queries
            k_proj: Projeção linear que produz as keys
            v_proj: Projeção linear que produz os values
            o_proj: Projeção linear aplicada após a concatenação das cabeças
        """
        super().__init__()
        if positional_encoder is None:
            warnings.warn(
                "Positional encoder não foi identificado",
                stacklevel=2,
            )

        assert d_model % num_heads == 0, (
            'O valor de "d_model" deve ser divisível pelo valor de "num_heads."'
        )

        self.head_dim = d_model // num_heads
        self.num_heads = num_heads

        self.positional_encoder = positional_encoder

        self.q_proj = Linear(
            d_model,
            self.head_dim * num_heads,
            device=device,
            dtype=dtype,
        )
        self.k_proj = Linear(
            d_model,
            self.head_dim * num_heads,
            device=device,
            dtype=dtype,
        )
        self.v_proj = Linear(
            d_model,
            self.head_dim * num_heads,
            device=device,
            dtype=dtype,
        )
        self.o_proj = Linear(
            self.head_dim * num_heads,
            d_model,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Float[
            torch.Tensor,
            "... sequence_length d_model",
        ],
        token_positions: Int[torch.Tensor, "... sequence_length"] | None = None,
        is_causal: bool = True,
        attn_mask: Bool[
            torch.Tensor,
            "... sequence_length sequence_length",
        ]
        | Float[
            torch.Tensor,
            "... sequence_length sequence_length",
        ]
        | None = None,
    ) -> Float[
        torch.Tensor,
        "... sequence_length d_model",
    ]:
        """
        Aplica MHA ao tensor de input.

        A entrada é projetada em queries, keys e values. Essas representações são separadas entre as cabeças e recebem, opcionalmente, codificação  posicional. Em seguida, a atenção por produto escalar escalado é
        calculada. As saídas das cabeças são concatenadas e projetadas novamente para a dimensão do modelo

        Args:
            x: Tensor de input com shape "(..., sequence_length, d_model)"
            token_positions: Posições associadas aos tokens, com shape (..., sequence_length). São utilizadas apenas pelo codificador posicional. Quando "None", o codificador utiliza posições sequenciais iniciadas em zero
            is_causal: Se "True", impede que cada posição consulte tokens futuros. Deve ser "False" quando "attn_mask" é fornecida.
            attn_mask: Máscara opcional compatível com os scores de atenção.

        Returns:
            Tensor com shape "(..., sequence_length, d_model)", contendo as
            representações contextualizadas.
        """
        query = einops.rearrange(
            self.q_proj(x),
            "... sequence_length (num_heads head_dim) -> ... num_heads sequence_length head_dim",
            num_heads=self.num_heads,
        )
        key = einops.rearrange(
            self.k_proj(x),
            "... sequence_length (num_heads head_dim) -> ... num_heads sequence_length head_dim",
            num_heads=self.num_heads,
        )
        value = einops.rearrange(
            self.v_proj(x),
            "... sequence_length (num_heads head_dim) -> ... num_heads sequence_length head_dim",
            num_heads=self.num_heads,
        )

        if self.positional_encoder is not None:
            if token_positions is not None:
                token_positions = einops.rearrange(
                    token_positions,
                    "... sequence_length -> ... 1 sequence_length",
                )

            query = self.positional_encoder(query, token_positions)
            key = self.positional_encoder(key, token_positions)

        attn = scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=is_causal,
            attn_mask=attn_mask,
        )
        attn = einops.rearrange(
            attn,
            "... num_heads sequence_length head_dim -> ... sequence_length (num_heads head_dim)",
        )
        output = self.o_proj(attn)

        return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        positional_encoder: RotaryPositionalEmbedding | None,
        eps: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Inicializa o blocos do Transformer

        O bloco contém duas subcamadas: um mecanismo de MHA e uma rede feed-forward SwiGLU. Cada subcamada possui o mecanismo de pré-norm usando o RMSNorm e a conexão residual

        Args:
            d_model: Dimensão das representações de input e output
            d_ff: Dimensão intermediária da rede feed-forward SwiGLU
            num_heads: Quantidade de cabeças utilizadas pelo MHA,"d_model" deve ser divisível por esse valor
            positional_encoder: Codificador posicional. Pode ser "None" para desativar a codificação posicional
            eps: Constante de estabilidade numérica utilizada pelas camadas RMSNorm
            device: Device no qual os parâmetros do bloco serão criados. Quando "None", utiliza o dispositivo padrão do PyTorch
            dtype: Type dos parâmetros. Quando "None", utiliza o dtype padrão do PyTorch

        Attributes:
            d_model: Dimensão das representações processadas pelo bloco
            d_ff: Dimensão intermediária da rede feed-forward
            num_heads: Quantidade de cabeças de atenção
            device: Device informado durante a inicialização
            mha: Módulo de MHA
            ffn: Rede feed-forward baseada em SwiGLU
            attn_norm: Pré norm usando a RMSNorm para a fase de atenção
            ffn_norm: Pré norm usando a RMSNorm para a fase de feed-forward

        """
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.device = device

        self.mha = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            positional_encoder=positional_encoder,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLU(
            dim=d_model,
            hidden_dim=d_ff,
            device=device,
            dtype=dtype,
        )
        self.attn_norm = RMSNorm(
            dim=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        )
        self.ffn_norm = RMSNorm(
            dim=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_model"],
        token_positions: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        """
        Processa a entrada pelas subcamadas de atenção e feed-forward

        A entrada é normalizada antes de cada subcamada. A saída de cada
        subcamada é somada à sua respectiva entrada por meio de uma conexão
        residual

        Args:
            x: Tensor de input com shape (..., sequence_length, d_model)
            token_positions: Posições dos tokens utilizadas pelo codificador
                posicional da atenção, com shape (..., sequence_length).

        Returns:
            Tensor contextualizado com o mesmo shape e dtype de "x".
        """
        x = x + self.mha(self.attn_norm(x), token_positions)
        x = x + self.ffn(self.ffn_norm(x))

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_layers: int,
        num_heads: int,
        vocab_size: int,
        context_length: int,
        theta: float,
        eps: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        """
        Inicializa um Transformer decoder-only

        O modelo converte identificadores de tokens em representações vetoriais, processa essas representações por uma sequência de blocos Transformer e projeta o resultado para o espaço do vocabulário

        Todos os blocos compartilham a mesma instância de "RotaryPositionalEmbedding". A atenção de cada bloco é causal, de modo que a representação de uma posição depende apenas dela mesma e das posições anteriores

        A saída contém logits não normalizados para cada token do vocabulário. Esses logits podem ser usados para calcular a cross-entropy durante o treinamento ou para selecionar o próximo token durante a inferência

        Args:
            d_model: Dimensão dos embeddings e das representações internas
            d_ff: Dimensão intermediária das redes feed-forward SwiGLU
            num_layers: Quantidade de blocos do Transformer
            num_heads: Quantidade de cabeças de atenção em cada bloco, "d_model" deve ser divisível por esse valor
            vocab_size: Quantidade de tokens distintos no vocabulário
            context_length: Quantidade máxima de posições suportadas pelo cache do RoPE
            theta: Base utilizada no cálculo das frequências do RoPE
            eps: Constante de estabilidade numérica utilizada pelas RMSNorm
            device: Device no qual os parâmetros serão criados. Quando "None', utiliza o dispositivo padrão do PyTorch
            dtype: Type dos parâmetros. Quando "None", utiliza o dtype padrão do PyTorch.

        Attributes:
            tok_embeddings: Embedding correspondente a cada token
            positional_encoder: RoPE compartilhado entre todos os blocos
            transformer_blocks: List contendo "num_layers" blocos Transformer
            norm: RMSNorm aplicada após o último bloco.
            output: Projeção linear de "d_model" -> "vocab_size"
        """

        self.tok_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        d_head = d_model // num_heads

        self.positional_encoder = RotaryPositionalEmbedding(
            context_length=context_length,
            dim=d_head,
            theta=theta,
            device=device,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_heads,
                    positional_encoder=self.positional_encoder,
                    eps=eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = RMSNorm(
            dim=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        )
        self.output = Linear(
            d_model,
            vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Int[torch.Tensor, "... seq_len"],
        token_positions: Int[torch.Tensor, "... seq_len"] | None = None,
    ) -> Float[torch.Tensor, "... seq_len vocab_size"]:
        """
        Calcula os logits do próximo token para cada posição da entrada.

        Quando "token_positions" não é fornecido, são criadas posições
        sequenciais de zero até "sequence_length - 1". Essas posições são
        compartilhadas entre todas as dimensões de batch

        Args:
            x: Tensor de ids de tokens com shape "(..., sequence_length)``. Os ids devem pertencer ao intervalo "[0, vocab_size - 1]"
            token_positions: Posições utilizadas pelo RoPE. Pode ter shape "(sequence_length,)" para posições compartilhadas entre o batch ou "(..., sequence_length)" para posições específicas por exemplo. Quando "None", utiliza posições sequenciais iniciadas em zero

        Returns:
            Tensor em "torch.float32" com shape "(..., sequence_length, vocab_size)", contendo logits não
            normalizados para cada token do vocabulário
        """
        if token_positions is None:
            seq_len = x.size(-1)
            token_positions = torch.arange(seq_len, device=x.device)

        h = self.tok_embeddings(x)

        for transformer_block in self.transformer_blocks:
            h = transformer_block(h, token_positions)

        h = self.norm(h)
        output = self.output(h).float()

        return output
