import math

import einops
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from functional import softmax


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

        var = 2 / (in_features + out_features)
        std = math.sqrt(var)
        tensor = torch.empty(out_features, in_features, device=device, dtype=dtype)
        nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-3 * std, b=3 * std)

        self.weights = nn.Parameter(tensor)

    def forward(self, x: Float[torch.Tensor, "... in_features"]) -> Float[torch.Tensor, "... out_features"]:
        """


        Args:
            x:
        """

        return einops.einsum(x, self.weights.T, "... in_features, in_features out_features -> ... out_features")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embeddings_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Inicializa o Módulo de Embedding.

        Args:
            num_embeddings:
            embeddings_dim:
            device:
            dtype:

        Atributos:
            num_embeddings:
            embeddings_dim:
            weight:
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim

        tensor = torch.empty(num_embeddings, embeddings_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=-3, b=3)

        self.weight = nn.Parameter(tensor)

    def forward(self, token_ids: Int[torch.LongTensor, "... seq_len"]) -> Float[torch.Tensor, "... seq_len embeddings_dim"]:
        """


        Args:
            token_ids:
        """

        return self.weight[token_ids]


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
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Float[torch.Tensor, "... dim"]) -> Float[torch.Tensor, "... dim"]:
        """


        Args:
            x:
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        result = (x / rms) * self.weight

        return result.to(in_dtype)


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

        self.w1 = Linear(dim, hidden_dim, device=device, dtype=dtype)
        self.w2 = Linear(dim, hidden_dim, device=device, dtype=dtype)
        self.w3 = Linear(hidden_dim, dim, device=device, dtype=dtype)

    def forward(self, x: Float[torch.Tensor, "... dim"]) -> Float[torch.Tensor, "... dim"]:
        """


        Args:
            x:
        """
        x1 = self.w1(x)
        values = x1 * torch.sigmoid(self.w1(x1))

        gates = self.w2(x)

        return self.w3(values * gates)


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
        x: Float[torch.Tensor, "... seq_len dim"],
        token_positions: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len dim"]:
        """


        Args:
            x:
            token_positions:
        """
        x1, x2 = x[..., 0::2], x[..., 1::2]
        x_rh = einops.rearrange(
            torch.stack([-x2, x1], dim=-1),
            "... seq_len half_dim pair -> ... seq_len (half_dim pair)",
        )

        cos = self.cos.to(x.dtype)
        sin = self.sin.to(x.dtype)

        x = x * cos[token_positions, :] + x_rh * sin[token_positions, :]

        return x


def scaled_dot_product_attention(
    query: Float[torch.Tensor, "... seq_len d_k"],
    key: Float[torch.Tensor, "... seq_len d_k"],
    value: Float[torch.Tensor, "... seq_len d_v"],
    is_causal: bool | None,
):
    """
    Aplica o Scaled Dot Product Attention (SDPA).

    Args:
        query:
        key:
        value:
        is_causal:
    """
    device, dtype = query.device, query.dtype

    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(key.size(-1))

    attn_bias = torch.zeros(L, S, dtype=dtype, device=device)
    if is_causal:
        temp_mask = torch.ones(L, S, dtype=bool, device=device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    attn = einops.einsum(
        query,
        key,
        "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k",
    )  # query @ key.T

    attn *= scale_factor
    attn += attn_bias
    attn = softmax(attn, dim=-1)
    attn = einops.einsum(attn, value, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")

    return attn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        is_causal: bool,
        theta: float,
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
            head_dim:
            rope:
            w_q:
            w_k:
            w_v:
            w_o:
        """
        super().__init__()

        self.head_dim = d_model // num_heads
        self.is_causal = is_causal

        self.rope = RotaryPositionalEmbedding(
            theta=theta,
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            device=device,
        )

        self.w_q = Linear(d_model, self.head_dim * num_heads, device=device, dtype=dtype)
        self.w_k = Linear(d_model, self.head_dim * num_heads, device=device, dtype=dtype)
        self.w_v = Linear(d_model, self.head_dim * num_heads, device=device, dtype=dtype)
        self.w_o = Linear(self.head_dim * num_heads, d_model, device=device, dtype=dtype)

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
        query = einops.rearrange(
            self.w_q(x),
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            head_dim=self.head_dim,
        ).contiguous()
        key = einops.rearrange(
            self.w_k(x),
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            head_dim=self.head_dim,
        ).contiguous()
        value = einops.rearrange(
            self.w_v(x),
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            head_dim=self.head_dim,
        ).contiguous()

        query = self.rope(query, token_positions)
        key = self.rope(key, token_positions)
        output = scaled_dot_product_attention(query, key, value, is_causal=self.is_causal)
        output = einops.rearrange(
            output,
            "... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)",
        )
        output = self.w_o(output)

        return output