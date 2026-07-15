import math

import einops
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
import warnings

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
        Inicializa o módulo Linear

        Sem bias, seguindo a prática comum em LLMs.

        Args:
            in_features:
            out_features:
            device:
            dtype:

        Atributos:
            in_features:
            out_features:
            weights:
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Glorot inicialization
        var = 2 / (in_features + out_features)
        std = math.sqrt(var)
        tensor = torch.empty(
            out_features, in_features, device=device, dtype=dtype
        )
        nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-3 * std, b=3 * std)

        self.weights = nn.Parameter(tensor, requires_grad=True)

    def forward(
        self, x: Float[torch.Tensor, "... in_features"]
    ) -> Float[torch.Tensor, "... out_features"]:
        """


        Args:
            x:
        """

        return einops.einsum(
            x,
            self.weights,
            "... in_features, out_features in_features-> ... out_features",
        )

    def extra_repr(self):
        return f"in_features={self.weight.shape[1]}, out_features={self.weights.shape[0]}"


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Inicializa o Módulo de Embedding.

        Args:
            num_embeddings: Número de embeddings de acordo com o tamanho do vocabulário.
            embedding_dim: Tamanho da dimensão de cada embedding.
            device:
            dtype:

        Atributos:
            num_embeddings:
            embedding_dim:
            weight:
        """
        super().__init__()
        tensor = torch.empty(
            num_embeddings, embedding_dim, device=device, dtype=dtype
        )
        nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=-3, b=3)

        self.weights = nn.Parameter(tensor, requires_grad=True)

    def forward(
        self, token_ids: Int[torch.LongTensor, "..."]
    ) -> Float[torch.Tensor, "... embedding_dim"]:
        """


        Args:
            token_ids:
        """

        return self.weights[token_ids, :]

    def extra_repr(self):
        return f"vocab_size={self.weight.shape[0]}, embedding_dim={self.weights.shape[1]}"


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Inicializa o Root Mean Squared Layer Normalization (RMSNorm).

        Args:
            theta:
            dim:
            eps:
            device:
            dtype:

        Atributos:
            weight:
            eps:
        """
        super().__init__()
        self.weights = nn.Parameter(
            torch.ones(dim, device=device, dtype=dtype), requires_grad=True
        )
        self.eps = eps

    def forward(
        self, x: Float[torch.Tensor, "... dim"]
    ) -> Float[torch.Tensor, "... dim"]:
        """


        Args:
            x:
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
        Inicializa a função de ativação Swish Linear Gated Unit (SwiGLU).

        Args:
            theta:
            dim:
            hidden_dim:
            device:
            dtype:

        Atributos:
            w1:
            w2:
            w3:
        """
        super().__init__()

        self.w1_weight = Linear(dim, hidden_dim, device=device, dtype=dtype)
        self.w2_weight = Linear(hidden_dim, dim, device=device, dtype=dtype)
        self.w3_weight = Linear(dim, hidden_dim, device=device, dtype=dtype)

    def forward(
        self, x: Float[torch.Tensor, "... dim"]
    ) -> Float[torch.Tensor, "... dim"]:
        """


        Args:
            x:
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
        Inicializa o Rotary Position Embedding (RoPE).

        Args:
            theta:
            dim:
            max_seq_len:
            device:

        Atributos:
            cos:
            sin:
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
        if dim % 2 != 0:
            raise ValueError(
                f'O valor de "dim" deve ser par, mas recebeu {dim}.'
            )

        pairs_counts = torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
        t = torch.arange(context_length, dtype=torch.float32, device=device)
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
        Args:
            x:
            token_positions:
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
        x2_rot = x2 * sin + x1 * cos

        result = torch.stack((x1_rot, x2_rot), dim=-1).flatten(-2)

        return result


def scaled_dot_product_attention(
    query: Float[torch.Tensor, "... queries d_k"],
    key: Float[torch.Tensor, "  ... keys    d_k"],
    value: Float[torch.Tensor, "... keys    d_v"],
    is_causal: bool,
    attn_mask: Bool[torch.Tensor, " ... queries keys"] | Float[torch.Tensor, "... queries keys"] | None = None,
) -> Float[torch.Tensor, "... queries d_v"]:
    """
    Esta função implementa o Scaled Dot Product Attention (SDPA).

    Args:
        query: Tensor de queries, pode ter qualquer número de dimensões iniciais.
        key: Tensor de keys, compartilha o número de dimensões iniciais com "query".
        value: Tensor de values,
        is_causal:
        attn_mask:

    Returns:
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

        causal_mask = torch.ones(queries, keys, dtype=torch.bool, device=query.device).triu(
            diagonal=1
        )
        attn_scores.masked_fill_(causal_mask, float("-inf"))

    if attn_mask is not None:
        if attn_mask.device != query.device:
            attn_mask = attn_mask.to(query.device)

        if attn_mask.dtype == torch.bool:
            attn_scores.masked_fill_(attn_mask.logical_not(), float("-inf"))
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
        eps: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Inicializa o Multi Head Attention (MHA).

        Args:
            Args:
            d_model:
            num_heads:
            max_seq_len:
            theta:
            device:
            dtype:

        Atributos:
            num_heads:
            rope:
            q_proj:
            k_proj:
            v_proj:
            o_proj:
        """
        super().__init__()
        if positional_encoder is None:
            warnings.warn(
                "Positional encoder não foi identificado", stacklevel=2
            )

        assert d_model % num_heads == 0, (
            'O valor de "d_model" deve ser divisível pelo valor de "num_heads."'
        )

        self.head_dim = d_model // num_heads
        self.num_heads = num_heads

        self.positional_encoder = positional_encoder # RoPE

        # QK-Norm
        self.q_norm = RMSNorm(
            dim=self.head_dim, eps=eps, device=device, dtype=dtype
        )
        self.k_norm = RMSNorm(
            dim=self.head_dim, eps=eps, device=device, dtype=dtype
        )

        self.q_proj = Linear(
            d_model, self.head_dim * num_heads, device=device, dtype=dtype
        )
        self.k_proj = Linear(
            d_model, self.head_dim * num_heads, device=device, dtype=dtype
        )
        self.v_proj = Linear(
            d_model, self.head_dim * num_heads, device=device, dtype=dtype
        )
        self.o_proj = Linear(
            self.head_dim * num_heads, d_model, device=device, dtype=dtype
        )

    def forward(
        self,
        x: Float[torch.Tensor, "... sequence_length d_model"],
        token_positions: Int[torch.Tensor, "... sequence_length"] | None = None,
        is_causal: bool = True,
        attn_mask: Bool[torch.Tensor, "... sequence_length sequence_length"] | Float[torch.Tensor, "... sequence_length sequence_length"] | None = None
    ) -> Float[torch.Tensor, "... sequence_length d_model"]:
        """


        Args:
            x:
            token_positions:
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

        query = self.q_norm(query)
        key = self.k_norm(key)

        if self.positional_encoder is not None:
            if token_positions is not None:
                token_positions = einops.rearrange(
                    token_positions,
                    "... sequence_length -> ... 1 sequence_length",
                )

            query = self.positional_encoder(query, token_positions)
            key = self.positional_encoder(key, token_positions)

        attn = scaled_dot_product_attention(
            query, key, value, is_causal=is_causal, attn_mask=attn_mask
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
        Inicializa o bloco do Transformer.

        Args:
            d_model:
            d_ff:
            num_heads:
            max_seq_len:
            theta:
            eps:
            device:
            dtype:

        Atributos:
            ffn:
            mha:
            attn_norm:
            ffn_norm
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
            eps=eps,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLU(
            dim=d_model, hidden_dim=d_ff, device=device, dtype=dtype
        )
        self.attn_norm = RMSNorm(
            dim=d_model, eps=eps, device=device, dtype=dtype
        )
        self.ffn_norm = RMSNorm(
            dim=d_model, eps=eps, device=device, dtype=dtype
        )

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_model"],
        token_positions: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        """


        Args:
            x:
            token_positions:
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
        """
        Inicializa o modelo de Transformer.

        Args:
            d_model:
            d_ff:
            num_heads:
            max_seq_len:
            theta:
            eps:
            device:
            dtype:

        Atributos:
            tok_embeddings:
            transformer_blocks:
            norm:
            output:
        """
        super().__init__()

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

        self.norm = RMSNorm(dim=d_model, eps=eps, device=device, dtype=dtype)
        self.output = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        x: Int[torch.Tensor, "... seq_len"],
        token_positions: Int[torch.Tensor, "... seq_len"] | None = None,
    ) -> Float[torch.Tensor, "... seq_len vocab_size"]:
        if token_positions is None:
            seq_len = x.size(-1)
            token_positions = torch.arange(seq_len, device=x.device)

        h = self.tok_embeddings(x)

        for transformer_block in self.transformer_blocks:
            h = transformer_block(h, token_positions)

        h = self.norm(h)
        output = self.output(h).float()

        return output
