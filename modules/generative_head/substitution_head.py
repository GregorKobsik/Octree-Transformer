import torch
import torch.nn as nn

from ..utils import Deconvolution, Linear


class SubstitutionHead(nn.Module):
    def __init__(self, num_vocab, embed_dim, spatial_dim, conv_size, **_):
        """ Performs a concolutional transformation from transformer latent space into target value logits.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different target token values (exclusive padding token '0').
            embded_dim: Dimension of the latent embedding space of the transformer.
            spatial_dim: Spatial dimension (2D/3D) of the sequence data.
            conv_size: Convolution kernel size and stride.
        """
        super(SubstitutionHead, self).__init__()
        self.embed_dim = embed_dim

        self.deconvolution_1 = Deconvolution(embed_dim, embed_dim // 2, conv_size)
        self.deconvolution_0 = Deconvolution(embed_dim // 2, embed_dim // 4, conv_size)
        self.linear = Linear(embed_dim // 4, num_vocab)

    def forward(self, x, value, depth, pos):
        """ Transforms the output of the transformer target value logits.

        Transforms one token of the latent vector into multiple tokens of the target vector through de-convolutional
        operations. In the case of a quadtree one token is responsible for up to 16 target tokens. In the case of a
        octree one token is responsible for up to 64 target tokens. Only tokens, which correspond to a mixed target
        value token in the penultimate layer are transformed into target sequence tokens.

        Args:
            x: Output of the transformer, the latent vector [N, T'', E].
            value: Value token sequence, with penultimate and last layer.
            depth: Depth token sequence, with penultimate and last layer.
            pos: Position token sequence, with penultimate and last layer.

        Return
            Logits of target value sequence.
        """
        batch_size = value.shape[0]

        # create intermediate tensor to hold values of second-last layer
        idx_1 = torch.argmax(depth, dim=1)
        val_1 = torch.zeros((batch_size, torch.max(idx_1)), device=x.device)  # [N, T'']

        # discard last layer of value tokens
        for i in range(batch_size):
            val_1[i, :idx_1[i]] = value[i, :idx_1[i]]

        # deconvolute the latent space - sequence length equals number of tokens in the penultimate layer
        y_1 = self.deconvolution_1(x)

        # create intermediate tensor to hold mixed tokens
        len_1 = torch.sum(val_1 == 2, dim=1)
        x_0 = torch.zeros((batch_size, int(torch.max(len_1)), self.embed_dim // 2), device=x.device)  # [N, T', C]

        # select only latent vectors, which correspond to mixed tokens in the penultimate layer
        for i in range(batch_size):
            # TODO: Check for value overflow
            # TODO: Handle overflow/clipped values in the embedding ...
            x_0[i, :len_1[i]] = y_1[i, val_1[i] == 2]  # [N, T', C]

        # deconvolute the intermediate latent space - create new tokens in latent space for each mixed token
        y_0 = self.deconvolution_0(x_0)  # [N, T, C]

        # compute logits of generated tokens
        return self.linear(y_0)  # [N, T, V]
