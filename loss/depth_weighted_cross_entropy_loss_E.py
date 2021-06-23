import torch.nn.functional as F

from torch.nn.modules.loss import CrossEntropyLoss

from torch import Tensor
from typing import Optional


class DepthWeightedCrossEntropyLossE(CrossEntropyLoss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = 0,
        spatial_dim: int = 1,
        max_depth: int = 5,
        **_,
    ) -> None:
        """ TODO: add description

        Args:
            weight:
            size_average:
            ignore_index:
            spatial_dim:
            max_depth:

        Return:

        """
        super(DepthWeightedCrossEntropyLossE, self).__init__(weight, size_average, ignore_index, None, 'none')
        self.ignore_index = ignore_index
        self.max_depth = max_depth
        self.spatial_dim = spatial_dim

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """ TODO: add description

        Args:
            logits: [N, T, V]
            target: [N, T]

        Return:

        """
        # unpack target sequence
        tgt_val, tgt_dep, _ = target  # [N, T]

        # flatten tensors
        input = logits.view(-1, logits.size(-1))  # [N*T, V]
        target = tgt_val.reshape(-1)  # [N*T]

        # compute loss
        loss = F.cross_entropy(
            input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction
        )  # [N*T]

        # compute bell curve shaped weighting - TODO: add parametrisation
        d_weight = tgt_dep.clone().detach()  # [N, T]
        d_weight[tgt_dep == 1] = 1
        d_weight[tgt_dep == 2] = 2
        d_weight[tgt_dep == 3] = 3
        d_weight[tgt_dep == 4] = 2
        d_weight[tgt_dep == 5] = 1

        # scale loss
        d_weight[tgt_dep == 0] = 0
        loss = loss * d_weight.view(-1)  # [N*T]

        # return loss
        return loss.reshape(tgt_val.shape)  # [N, T]
