import torch.nn.functional as F

from torch.nn import CrossEntropyLoss

from torch import Tensor
from typing import Optional


class CrossEntropyLoss(CrossEntropyLoss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = 0,
        reduce=None,
        reduction='mean',
    ) -> None:
        super(CrossEntropyLoss, self).__init__(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(
            input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction
        )
