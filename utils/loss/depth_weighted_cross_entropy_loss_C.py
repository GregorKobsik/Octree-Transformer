import torch
import torch.nn.functional as F

from torch.nn.modules.loss import CrossEntropyLoss

from torch import Tensor
from typing import Optional


class DepthWeightedCrossEntropyLossC(CrossEntropyLoss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = 0,
        spatial_dim: int = 1,
        max_depth: int = 5,
        **_,
    ) -> None:
        """ Defines a weighted cross entropy loss function based on the depth of each layer.

        This criterion combines LogSoftmax and NLLLoss in one single class. The weighting of the loss is done
        expotentially based on the depth of the token: fx(d) = `spatial_dim`^(`max_depth` - d).

        Args:
            weight: Manual rescaling weight given to each class. If given, has to be a Tensor of size V.
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
            spatial_dim: Defines the spatial dimension of the data used in the model.
            max_depth: Defines the maximum depth value used in the model.
        """
        super(DepthWeightedCrossEntropyLossC, self).__init__(weight, size_average, ignore_index, None, 'none')
        self.ignore_index = ignore_index
        self.max_depth = max_depth
        self.spatial_dim = spatial_dim

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """ Computes the weighted cross entropy loss given logits and targets.

        Args:
            logits: Contains raw, unnormalized scores for each class [N, T, V].
            target: Is a tuple of target sequences for value, depth and position with each a shape of [N, T].

        Return:
            Loss for each token with the shape [N, T].
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

        # compute expotentially growing inverse depth weighting
        d_weight = 0.4**tgt_dep  # [N, T]

        # scale loss
        d_weight = torch.where(tgt_dep != 0, d_weight, torch.zeros_like(d_weight))  # [N, T]
        loss = loss * d_weight.view(-1)  # [N*T]

        # return loss
        return loss.reshape(tgt_val.shape)  # [N, T]
