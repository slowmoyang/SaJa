import torch
from torch import Tensor
from torch import nn

class ObjWise(nn.Module):
    """
    torch.nn.Sequential-like container class to enable the element-wise
    operation, using the data location information. data means non-pad part in
    a batch.
    """
    def __init__(self, *operation) -> None:
        super(ObjWise, self).__init__()

        if len(operation) == 1:
            operation = operation[0]
        elif len(operation) > 1:
            operation = nn.Sequential(*operation)
        else:
            raise ValueError

        self.operation = operation

    def forward(self, input: Tensor, data_mask: Tensor) -> Tensor:
        '''
        Args:
            input: (batch_size, seq_len, input_size)
            data_mask: (batch_size, seq_len)
        Returns:
            input: (batch_size, seq_len, output_size)
        '''
        batch_size, seq_len, input_size = input.shape
        dtype, device = input.dtype, input.device

        select_mask = data_mask.reshape(-1, 1)

        input = input.reshape(-1, input_size)
        input = input.masked_select(select_mask)
        input = input.reshape(-1, input_size)

        output_source = self.operation(input)
        output_size = output_source.size(1)

        scatter_mask = select_mask.expand(select_mask.size(0), output_size)
        output = torch.zeros((batch_size * seq_len, output_size),
                             dtype=dtype, device=device)
        output = output.masked_scatter(mask=scatter_mask, source=output_source)
        output = output.reshape(batch_size, seq_len, output_size)
        return output
