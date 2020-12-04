from typing import Tuple
import torch
from torch import Tensor
from torch import nn
from torch.nn import init
import torch.nn.functional as F


class Attention(nn.Module):
    """Multi-Head Scaled Dot Product Attention, which can take a batch with
    variable-length sequences (or 'sets'). torch.nn.MultiheadAttention (v.1.7.0)
    can take only `key_padding_mask` with the shape of (N, S) and `attn_mask`
    with the shape of (L, S) and is not suitable for processing multi-events
    while using all-jets in each event.
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 num_heads: int,
                 dropout_prob: float = 0.1) -> None:
        super(Attention, self).__init__()
        assert output_size % num_heads == 0

        self.input_size = input_size,
        self.output_size = output_size
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

        self.depth = int(output_size / num_heads)
        self.scale_factor = self.depth ** -0.5

        self.linear_key = nn.Linear(input_size, output_size, bias=False)
        self.linear_value = nn.Linear(input_size, output_size, bias=False)
        self.linear_query = nn.Linear(input_size, output_size, bias=False)
        self.linear_output = nn.Linear(output_size, output_size, bias=False)
        self.dropout = nn.Dropout(p=dropout_prob)

        self.reset_parameters()
 
    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.linear_key.weight)
        init.xavier_uniform_(self.linear_value.weight)
        init.xavier_uniform_(self.linear_query.weight)
        init.xavier_uniform_(self.linear_output.weight)

    def forward(
            self,
            key: Tensor,
            value: Tensor,
            query: Tensor,
            pad_mask_key: Tensor,
            pad_mask_query: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        '''
        Args:
            key: (B, S, D)
            value: (B, S, D)
            query: (B, T, D)
        Returns:
        '''
        assert key.size(1) == value.size(1)
        assert key.size(2) == query.size(2)
        # NOTE I'm testing `_forward` to enable jitted SaJa in the future.
        return self._forward(key, value, query, pad_mask_key, pad_mask_query)

    @torch.jit.export
    def _forward(
            self,
            key: Tensor,
            value: Tensor,
            query: Tensor,
            pad_mask_key: Tensor,
            pad_mask_query: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        '''
        Args:
            query: (batch, target_len, input_size)
            key: (batch, source_len, input_size)
            value: (batch, source_len, input_size)
            pad_mask_query: (batch, target_len)
            pad_mask_key: (batch, source_len)
        '''
        batch_size, source_len, _ = key.size()
        target_len = query.size(1)

        key = self.linear_key(key)
        value = self.linear_value(value)
        query = self.linear_query(query)

        key = self.split(key)
        query = self.split(query)
        value = self.split(value)

        key = key.contiguous().permute(0, 2, 1)
        query = self.scale_factor * query

        attention_logits = torch.bmm(query, key)

        pad_mask_key = pad_mask_key.unsqueeze(1)
        pad_mask_key = pad_mask_key.expand(-1, target_len, -1)
        pad_mask_key = pad_mask_key.unsqueeze(1)
        pad_mask_key = pad_mask_key.expand(-1, self.num_heads, -1, -1)
        pad_mask_key = pad_mask_key.reshape(batch_size * self.num_heads, target_len, source_len)
        attention_logits = attention_logits.masked_fill(pad_mask_key, float('-inf'))

        attention = attention_logits.softmax(dim=2)
        attention = self.dropout(attention)

        pad_mask_query = pad_mask_query.unsqueeze(2)
        pad_mask_query = pad_mask_query.unsqueeze(1)
        pad_mask_query = pad_mask_query.expand(-1, self.num_heads, -1, -1)
        pad_mask_query = pad_mask_query.reshape(batch_size * self.num_heads, target_len, 1)
        attention = attention.masked_fill(pad_mask_query, 0)

        output = torch.bmm(attention, value)
        output = self.combine(output)
        output = self.linear_output(output)

        attention = attention.reshape(batch_size, self.num_heads, target_len, source_len)

        return (output, attention)

    @torch.jit.export
    def split(self, tensor: Tensor) -> Tensor:
        '''Split Q, K and V into multiple heads.
        Args:
            tensor: (batch, length, dim), where dim = head * depth.
        Returns:
            tensor: (batch * head, length, depth)
        '''
        batch_size, seq_len, _, = tensor.shape

        tensor = tensor.reshape(batch_size, seq_len, self.num_heads, self.depth)
        tensor = tensor.contiguous().permute(0, 2, 1, 3)
        tensor = tensor.reshape(batch_size * self.num_heads, seq_len, self.depth)
        return tensor

    @torch.jit.export
    def combine(self, tensor: Tensor) -> Tensor:
        '''
        Args:
            tensor: (batch * head, length, depth)
        Returns:
            tensor: (batch, length, head * depth)
        '''
        seq_len = tensor.size(1)

        tensor = tensor.reshape(-1, self.num_heads, seq_len, self.depth)
        tensor = tensor.contiguous().permute(0, 2, 1, 3)

        batch_size = tensor.size(0)
        return tensor.reshape(batch_size, seq_len, self.output_size)

class SelfAttention(Attention):
    """
    """
    def forward(self, input: Tensor, pad_mask: Tensor) -> Tuple[Tensor, Tensor]:
        return self._forward(input, input, input, pad_mask, pad_mask)
