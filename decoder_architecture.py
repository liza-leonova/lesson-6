"""
Модуль с компонентами архитектуры Transformer.
Содержит реализации слоев модели.
"""

import torch
import math
from copy import deepcopy
from typing import Optional


class PositionEncoder(torch.nn.Module):
    """Класс позиционного кодирования"""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(2 * torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return x


class TokenEmbedder(torch.nn.Module):
    """Класс для эмбеддинга токенов"""

    def __init__(self, d_model: int, vocab_size: int, pad_idx: int) -> None:
        super().__init__()
        self.embedding_dim = d_model
        self.token_embedding = torch.nn.Embedding(vocab_size, self.embedding_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionEncoder(self.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        return x


class Attention(torch.nn.Module):
    """Класс механизма внимания"""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.embedding_dim = d_model
        self.scale_factor = math.sqrt(self.embedding_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale_factor
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        attention_weights = torch.nn.Softmax(dim=-1)(scores)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights


class MultiHeadAttention(torch.nn.Module):
    """Класс многоголового внимания"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding_dim = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)

        self.attention = Attention(d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch = q.size(0)
        q = self.q_proj(q).view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        x, attn_weights = self.attention(q, k, v, mask)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.embedding_dim)
        x = self.out_proj(x)
        x = self.dropout(x)
        return x, attn_weights


class FeedForwardNetwork(torch.nn.Module):
    """Класс feed-forward сети"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding_dim = d_model
        self.ffn_dim = d_ff

        if d_ff % d_model:
            raise ValueError(f"Размерность FFN {d_ff} должна делиться на размерность модели {d_model}")

        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.linear2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.linear2(self.dropout(x))
        return x


class DecoderBlock(torch.nn.Module):
    """Блок декодера Transformer"""

    def __init__(self, attention: MultiHeadAttention, ffn: FeedForwardNetwork, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention_layer = deepcopy(attention)
        self.ffn_layer = deepcopy(ffn)
        self.norm1 = torch.nn.LayerNorm(attention.embedding_dim)
        self.norm2 = torch.nn.LayerNorm(attention.embedding_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x_norm = self.norm1(x)
        x = x + self.attention_layer(x_norm, x_norm, x_norm, mask)[0]
        x_norm = self.norm2(x)
        x = self.dropout(x_norm + self.ffn_layer(x))
        return x


class DecoderStack(torch.nn.Module):
    """Стек блоков декодера"""

    def __init__(self, decoder_block: DecoderBlock, num_layers: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([deepcopy(decoder_block) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x