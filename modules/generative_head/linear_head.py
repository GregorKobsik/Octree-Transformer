import torch.nn as nn

from ..utils import Linear


class LinearHead(nn.Module):
    def __init__(self, spatial_encoding, num_vocab, embed_dim, head_dim, n_layer, **_):
        """ Performs a linear transformation from transformer latent space into target value logits.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different target token values (exclusive padding token '0').
            embded_dim: Dimension of the latent embedding space of the transformer.
            head_dim: Size of embedding dimensions used in the head layers.
            n_layer: Number of layers used in each linear or convolution block.
        """
        super(LinearHead, self).__init__()

        if n_layer > 1:
            linear = [nn.GELU(), nn.Linear(embed_dim, head_dim)]
            for i in range(n_layer - 2):
                linear += [nn.GELU(), nn.Linear(head_dim, head_dim)]
            linear += [nn.GELU(), Linear(head_dim, num_vocab)]
            self.linear = nn.Sequential(*linear)
        else:
            self.linear = Linear(embed_dim, num_vocab)

        # quick fix as this is the only place, where the output should be embed_dim instead of head_dim
        if spatial_encoding is not None:
            self.spatial_encoding = nn.Sequential(
                spatial_encoding,
                nn.Linear(head_dim, embed_dim)
            )
        else:
            self.spatial_encoding = None

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
        # add spatial decoding if available
        if self.spatial_encoding is not None:
            x = x + self.spatial_encoding(pos)

        return self.linear(x)
