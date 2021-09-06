import torch
import torch.nn.functional as F

from torch.nn.modules.loss import _WeightedLoss

from torch import Tensor
from typing import Optional


class DepthWeightedCrossEntropyLoss(_WeightedLoss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        ignore_index: int = 0,
        basis=1.0,
        **_,
    ) -> None:
        """ Defines a weighted cross entropy loss function based on the depth of each layer.

        This criterion combines LogSoftmax and NLLLoss in one single class. Each token is scaled with basis^depth.

        Args:
            weight: Manual rescaling weight given to each class. If given, has to be a Tensor of size V.
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
            basis: Defines the basis of the weighting function
        """
        super(DepthWeightedCrossEntropyLoss, self).__init__(weight, None, None, 'none')
        self.ignore_index = ignore_index
        self.basis = basis

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """ Computes the weighted cross entropy loss given logits and targets.

        Args:
            logits: Contains raw, unnormalized scores for each class [N, T, V].
            target: Is a tuple of target sequences for value, depth and position with each a shape of [N, T].

        Return:
            Loss for each token with the shape [N, T].
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

        # compute expotentially decreasing weighting
        d_weight = self.basis**tgt_dep  # [N, T]
        d_weight /= torch.mean(d_weight)

        # scale loss
        d_weight = torch.where(tgt_dep != 0, d_weight, torch.zeros_like(d_weight))  # [N, T]
        loss = loss * d_weight.view(-1)  # [N*T]

        # return loss
        return loss.reshape(tgt_val.shape)  # [N, T]
