import math

import einops
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int

from languagemodel.functional import softmax

# Tornar o RoPE injetável, pois um único objeto  pode ser usado em todas as leyers, tornado desnecessário o instanciamento dela em cada layer.

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
        return f"in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}"


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
        return f"vocab_size={self.weight.shape[0]}, embedding_dim={self.weight.shape[1]}"


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
        max_seq_len: int,
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

        pairs_counts = torch.arange(0, dim // 2, device=device)
        t = torch.arange(max_seq_len, device=device)
        inv_freq = theta ** ((-2 * pairs_counts) / dim)
        freqs = torch.outer(t, inv_freq)

        cos, sin = freqs.cos(), freqs.sin()

        self.register_buffer(
            "cos",
            torch.repeat_interleave(cos, repeats=2, dim=-1),
            persistent=False,
        )
        self.register_buffer(
            "sin",
            torch.repeat_interleave(sin, repeats=2, dim=-1),
            persistent=False,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "... sequence_length dim"],
        token_positions: Int[torch.Tensor, "... sequence_length"],
    ) -> Float[torch.Tensor, "... sequence_length dim"]:
        """


        Args:
            x:
            token_positions:
        """
        x1, x2 = x[..., 0::2], x[..., 1::2]
        x_rh = einops.rearrange(
            torch.stack([-x2, x1], dim=-1),
            "... sequence_length half_dim pair -> ... sequence_length (half_dim pair)",
        )

        cos = self.cos.to(x.dtype)
        sin = self.sin.to(x.dtype)

        x = x * cos[token_positions, :].unsqueeze(-3) + x_rh * sin[
            token_positions, :
        ].unsqueeze(-3)

        return x

    def extra_repr(self):
        return f"context_length={self._freq_cis_cache.shape[0]}, dim/2={self._freq_cis_cache.shape[1]}"

def scaled_dot_product_attention(
    query: Float[torch.Tensor, "... queries d_k"],
    key: Float[torch.Tensor, "  ... keys    d_k"],
    value: Float[torch.Tensor, "... keys    d_v"],
    is_causal: bool,
    attn_mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
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
    device, dtype = query.device, query.dtype

    queries, keys = query.size(-2), key.size(-2)
    d_k = key.size(-1)
    scale_factor = 1 / math.sqrt(d_k)

    attn_bias = torch.zeros((queries, keys), dtype=dtype, device=device)

    if is_causal:
        assert attn_mask is None, "attn_mask não pode ser passado junto com is_causal=True"

        temp_mask = torch.ones_like(attn_bias, dtype=torch.bool).triu_(
            diagonal=1
        )
        attn_bias.masked_fill_(temp_mask, float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_bias + attn_mask

    attn_scores = (
        einops.einsum(
            query,
            key,
            "... queries d_k, ... keys d_k -> ... queries keys",
        )
        * scale_factor
    )
    attn_scores += attn_bias
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
        max_seq_len: int,
        theta: float,
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

        assert d_model % num_heads == 0, 'O valor de "d_model" deve ser divisível pelo valor de "num_heads."'

        self.head_dim = d_model // num_heads
        self.num_heads = num_heads

        self.rope = RotaryPositionalEmbedding(
            theta=theta,
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            device=device,
        )

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
        token_positions: Int[torch.Tensor, "... sequence_length"],
        is_causal: bool = True,
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
        ).contiguous()
        key = einops.rearrange(
            self.k_proj(x),
            "... sequence_length (num_heads head_dim) -> ... num_heads sequence_length head_dim",
            num_heads=self.num_heads,
        ).contiguous()
        value = einops.rearrange(
            self.v_proj(x),
            "... sequence_length (num_heads head_dim) -> ... num_heads sequence_length head_dim",
            num_heads=self.num_heads,
        ).contiguous()

        query = self.q_norm(query)
        key = self.k_norm(key)

        query = self.rope(query, token_positions)
        key = self.rope(key, token_positions)

        attn = scaled_dot_product_attention(
            query, key, value, is_causal=is_causal
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
        max_seq_len: int,
        theta: float,
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
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.device = device

        self.mha = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
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

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_heads,
                    max_seq_len=context_length,
                    theta=theta,
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
