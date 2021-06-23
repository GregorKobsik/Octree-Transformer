import torch.nn.functional as F

from torch.nn.modules.loss import _WeightedLoss

from torch import Tensor
from typing import Optional


class CrossEntropyLoss(_WeightedLoss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        ignore_index: int = 0,
        **_,
    ) -> None:
        """ Defines a basic cross entropy loss function.

        This criterion combines LogSoftmax and NLLLoss in one single class.

        Args:
            weight: Manual rescaling weight given to each class. If given, has to be a Tensor of size V.
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
        """
        super(CrossEntropyLoss, self).__init__(weight, None, None, 'none')
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """ Computes the cross entropy loss given logits and targets.

        Args:
            logits: Contains raw, unnormalized scores for each class [N, T, V].
            target: Is a tuple of target sequences for value, depth and position with each a shape of [N, T].

        Return:
            Loss for each token with the shape [N, T].
        """
        # unpack target sequence
        tgt_val, _, _ = target  # [N, T]

        # flatten tensors
        input = logits.view(-1, logits.size(-1))  # [N*T, V]
        target = tgt_val.reshape(-1)  # [N*T]

        # compute cross entropy loss
        loss = F.cross_entropy(
            input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction
        )

        # return loss
        return loss.reshape(tgt_val.shape)  # [N, T]
