from typing import Tuple
from typing import Optional
import torch
from torch import Tensor
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from saja.modules.attention import SelfAttention
from saja.modules.objwise import ObjWise

class SelfAttentionBlock(nn.Module):
    """
    """
    def __init__(
            self,
            input_size: int,
            num_heads: int,
            filter_size: int,
            dropout_rate: float = 0.1,
            output_size: Optional[int] = None,
    ) -> None:
        super(SelfAttentionBlock, self).__init__()
        if output_size is None:
            output_size = input_size

        self.input_size = input_size
        self.num_heads = num_heads
        self.filter_size = filter_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        self.attention = SelfAttention(
            input_size=input_size,
            output_size=input_size,
            num_heads=num_heads)
        self.dropout_attention = nn.Dropout(p=dropout_rate)
        self.layer_norm_attention = ObjWise(nn.LayerNorm(input_size))
        self.ffn = ObjWise(
            nn.Linear(input_size, filter_size, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(filter_size, output_size, bias=True),
            nn.LeakyReLU())
        self.dropout_ffn = nn.Dropout(p=dropout_rate)
        self.layer_norm_ffn = ObjWise(nn.LayerNorm(output_size))

    def forward(self, x: Tensor, data_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """ TODO
        Args:
            x:
            data_mask:
        Returns:
            ouptut:
            attention_weight
        """
        pad_mask = torch.logical_not(data_mask)

        identity = x
        residual, attention = self.attention(x, pad_mask)
        residual = self.dropout_attention(residual)
        x = identity + residual
        x = self.layer_norm_attention(x, data_mask)

        identity = x
        residual = self.ffn(x, data_mask)
        residual = self.dropout_ffn(residual)
        x = identity + residual
        x = self.layer_norm_ffn(x, data_mask)

        return x, attention
