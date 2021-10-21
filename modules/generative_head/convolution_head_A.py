import torch.nn as nn

from ..utils import Deconvolution, Linear


class ConvolutionHeadA(nn.Module):
    def __init__(self, num_vocab, embed_dim, spatial_dim, conv_size, **_):
        """ Performs a convolutional transformation from transformer latent space into target value logits.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different target token values (exclusive padding token '0').
            embded_dim: Dimension of the latent embedding space of the transformer.
            spatial_dim: Spatial dimension (2D/3D) of the sequence data.
            conv_size: Convolution kernel size and stride.
        """
        super(ConvolutionHeadA, self).__init__()

        self.deconvolution = Deconvolution(embed_dim, embed_dim, conv_size)
        self.linear = Linear(embed_dim, num_vocab)

    def forward(self, x, value, depth, pos):
        """ Transforms the output of the transformer target value logits.

        Args:
            x: Output of the transformer, the latent vector [N, T', E].
            value: Target value token sequence [N, T].
            depth: Target depth token sequence [N, T].
            pos: Target position token sequence [N, T, A].

        Return
            Logits of target value sequence.
        """
        # deconvolute the latent space - create new tokens
        x = self.deconvolution(x)  # [N, T, E]

        # compute logits for each token
        return self.linear(x)  # [N, T, V]
