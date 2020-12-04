from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

def object_wise_cross_entropy(
        input: Tensor,
        target: Tensor,
        mask: Tensor,
        length: Tensor,
        class_last: bool = True,
        reduction: str = 'mean'
) -> Tensor:
    """Object-wise cross-entropy loss
    Args:
        input: (B, L, C)
        target:
        mask: 
        length:
        class_last:
        reduction:
    Returns:
        output:
    """
    if class_last:
        input = input.permute(0, 2, 1)
    loss = F.cross_entropy(input, target, reduction='none')
    loss.masked_fill_(mask, 0)

    length = length.to(input.dtype)
    loss = loss.sum(dim=1) / length

    if reduction == 'mean':
        loss = loss.mean()
    return loss

def saja_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        length: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        class_last: bool = True,
        reduction: str = 'mean',
) -> Tensor:
    """ Loss for Zero-Permutation Jet-Parton Assignment for fully-hadronic ttbar
    events.

    Args:
        input:
        target:
        length:
        mask:
        class_last:
        reduction:
    Returns:
    """
    # FIXME crude way
    # (QCD, b1, W1, b2, W2)
    permuted_target = target.clone()
    permuted_target[target == 1] = 3
    permuted_target[target == 2] = 4
    permuted_target[target == 3] = 1
    permuted_target[target == 4] = 2

    loss1 = object_wise_cross_entropy(input, target, mask, length, class_last,
                                      reduction='none')
    loss2 = object_wise_cross_entropy(input, permuted_target, mask, length,
                                      class_last, reduction='none')

    loss = torch.stack([loss1, loss2], dim=1)
    loss, _ = loss.min(dim=1)
    if reduction == 'mean':
        loss = loss.mean()
    return loss
