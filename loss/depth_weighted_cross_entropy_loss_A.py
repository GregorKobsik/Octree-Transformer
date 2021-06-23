import torch
import torch.nn.functional as F

from torch.nn.modules.loss import _WeightedLoss

from torch import Tensor
from typing import Optional


class DepthWeightedCrossEntropyLossA(_WeightedLoss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        ignore_index: int = 0,
        max_depth: int = 5,
        **_,
    ) -> None:
        """ TODO: add description

        Args:
            weight:
            size_average:
            ignore_index:
            max_depth:

        Return:

        """
        super(DepthWeightedCrossEntropyLossA, self).__init__(weight, None, None, 'none')
        self.ignore_index = ignore_index
        self.max_depth = max_depth

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """ TODO: add description

        Args:
            logits: [N, T, V]
            target: [N, T]

        Return:

        """
        # unpack target sequence
        tgt_val, tgt_dep, _ = target

        # flatten tensors
        input = logits.view(-1, logits.size(-1))  # [N*T, V]
        target = tgt_val.reshape(-1)  # [N*T]

        # compute loss
        loss = F.cross_entropy(
            input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction
        )  # [N*T]

        # compute linearily growing inverse depth weighting
        d_weight = 1 + self.max_depth - tgt_dep  # [N, T]

        # scale loss
        d_weight = torch.where(tgt_dep != 0, d_weight, torch.zeros_like(d_weight))  # [N, T]
        loss = loss * d_weight.view(-1)  # [N*T]

        # return loss
        return loss.reshape(tgt_val.shape)  # [N, T]
