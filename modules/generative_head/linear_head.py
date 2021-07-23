import torch.nn as nn

from ..utils import Linear


class LinearHead(nn.Module):
    def __init__(self, num_vocab, embed_dim, **_):
        """ Performs a linear transformation from transformer latent space into target value logits.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different target token values (exclusive padding token '0').
            embded_dim: Dimension of the latent embedding space of the transformer.
        """
        super(LinearHead, self).__init__()

        self.linear = Linear(embed_dim, num_vocab)

    def forward(self, x, value, depth, pos):
        """ Transforms the output of the transformer target value logits.

        Args:
            x: Output of the transformer, the latent vector [N, T, E].
            value: Target value token sequence [N, T].
            depth: Target depth token sequence [N, T].
            pos: Target position token sequence [N, T, A].

        Return
            Logits of target value sequence.
        """
        return self.linear(x)
