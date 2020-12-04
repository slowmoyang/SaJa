import torch
from torch import nn

from saja.modules import SelfAttentionBlock
from saja.modules import ObjWise

class SaJa(nn.Module):
    def __init__(self,
                 dim_input: int,
                 dim_ffn: int = 1024,
                 num_blocks: int = 6,
                 num_heads: int = 10,
                 depth: int = 32,
                 dropout_rate: float = 0.1,
                 dim_output: int = 5,
                 return_attention: bool = False) -> None:
        """
        """
        super(SaJa, self).__init__()
        self.dim_input = dim_input
        self.dim_ffn = dim_ffn
        self.num_blocks = num_blocks
        self.depth = depth
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.dim_output = dim_output
        self.return_attention = return_attention

        self.dim_model = num_heads * depth

        self.ffn_bottom = ObjWise(
            nn.Linear(dim_input, dim_ffn, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffn, self.dim_model, bias=True),
            nn.LeakyReLU())

        attention_blocks = []
        for _ in range(num_blocks):
            block = SelfAttentionBlock(self.dim_model, num_heads, dim_ffn,
                                       dropout_rate)
            attention_blocks.append(block)
        self.attention_blocks = nn.ModuleList(attention_blocks)

        self.ffn_top = ObjWise(
            nn.Linear(self.dim_model, dim_ffn, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_ffn, dim_output, bias=True))
            
    def forward(self, x, mask):
        x = self.ffn_bottom(x, mask)
        attention_list = []
        for block in self.attention_blocks:
            x, attention = block(x, mask)
            attention_list.append(attention)
        x = self.ffn_top(x, mask)
        attention_list = torch.stack(attention_list, dim=1)
        return (x, attention_list)
