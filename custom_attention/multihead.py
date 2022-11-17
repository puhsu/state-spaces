import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, softmax
from torch.nn import Linear, Module

logger = logging.getLogger(__name__)

"""https://github.com/sooftware/attentions/blob/master/attentions.py"""

try:
    from flash_attn import FlashAttention  # see flash-attention/flash_attn/flash_attention.py
    Attention = FlashAttention
except ImportError:
    logger.warning('Cannot import flash attention! Run `python setup.py install` in flash_attention directory!')


    class ScaledDotProductAttention(Module):
        """
        Scaled Dot-Product Attention proposed in "Attention Is All You Need"
        Compute the dot products of the query with all keys, divide each by sqrt(dim),
        and apply a softmax function to obtain the weights on the values
        Args: dim, mask
            dim (int): dimention of attention
            mask (torch.Tensor): tensor containing indices to be masked
        Inputs: query, key, value, mask
            - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
            - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
            - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
            - **mask** (-): tensor containing indices to be masked
        Returns: context, attn
            - **context**: tensor containing the context vector from attention mechanism.
            - **attn**: tensor containing the attention (alignment) from the encoder outputs.
        """

        def __init__(self, dim: int):
            super(ScaledDotProductAttention, self).__init__()
            self.sqrt_dim = np.sqrt(dim)

        def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
            score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

            if mask is not None:
                score.masked_fill_(mask.view(score.size()), -float('Inf'))

            attn = softmax(score, -1)
            context = torch.bmm(attn, value)
            return context, attn


    FlashAttention = None
    Attention = ScaledDotProductAttention


class MultiHeadAttention(Module):
    """
    Multi-Head Attention proposed in "Attention Is All You Need"
    Instead of performing a single attention function with d_model-dimensional keys, values, and queries,
    project the queries, keys and values h times with different, learned linear projections to d_head dimensions.
    These are concatenated and once again projected, resulting in the final values.
    Multi-head attention allows the model to jointly attend to information from different representation
    subspaces at different positions.
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
        where head_i = Attention(Q · W_q, K · W_k, V · W_v)
    Args:
        d_model (int): The dimension of keys / values / quries (default: 512)
        num_heads (int): The number of attention heads. (default: 8)
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): In transformer, three different ways:
            Case 1: come from previoys decoder layer
            Case 2: come from the input embedding
            Case 3: come from the output embedding (masked)
        - **key** (batch, k_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)
        - **value** (batch, v_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)
        - **mask** (-): tensor containing indices to be masked
    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features.
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.scaled_dot_attn = Attention(self.d_head)
        self.query_proj = Linear(d_model, self.d_head * num_heads)
        self.key_proj = Linear(d_model, self.d_head * num_heads)
        self.value_proj = Linear(d_model, self.d_head * num_heads)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)  # BxQ_LENxNxD
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)      # BxK_LENxNxD
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)  # BxV_LENxNxD

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxQ_LENxD
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)      # BNxK_LENxD
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxV_LENxD

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # BxNxQ_LENxK_LEN

        context, attn = self.scaled_dot_attn(query, key, value, attn_mask)

        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)  # BxTxND

        return context, attn
