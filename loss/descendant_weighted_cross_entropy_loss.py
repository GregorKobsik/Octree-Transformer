import torch
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss

from torch import Tensor
from typing import Optional


class DescendantWeightedCrossEntropyLoss(CrossEntropyLoss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        spatial_dim: int = 1,
    ) -> None:
        super(DescendantWeightedCrossEntropyLoss, self).__init__(weight, size_average, ignore_index, None, 'none')
        self.spatial_dim = spatial_dim

    def forward(self, input: Tensor, target: Tensor, depth: Tensor) -> Tensor:
        loss = F.cross_entropy(
            input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction
        )
        d_weight = torch.where(depth != 0, torch.max(depth) - depth + 1, 0)
        return torch.mean(loss * d_weight)
