import torch
import torch.nn as nn

from ..utils import Deconvolution, Linear


class SubstitutionHead(nn.Module):
    def __init__(self, spatial_encoding, num_vocab, embed_dim, spatial_dim, conv_size, **_):
        """ Performs a substitution transformation from transformer latent space into target value logits.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different target token values (exclusive padding token '0').
            embded_dim: Dimension of the latent embedding space of the transformer.
            spatial_dim: Spatial dimension (2D/3D) of the sequence data.
            conv_size: Convolution kernel size and stride.
        """
        super(SubstitutionHead, self).__init__()
        self.embed_dim = embed_dim

        self.deconvolution_1 = Deconvolution(embed_dim, embed_dim, conv_size)
        self.deconvolution_0 = Deconvolution(embed_dim, embed_dim, conv_size)
        self.linear = Linear(embed_dim, num_vocab)
        self.spatial_encoding = spatial_encoding

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
        max_depth = torch.max(depth)
        len_1 = torch.sum(depth == (max_depth - 1), dim=1)

        # create intermediate list to hold values
        val_1 = torch.zeros((batch_size, torch.max(len_1)), device=value.device)

        # splitt input in second-last (1) layer
        for i in range(batch_size):
            val_1[i, :len_1[i]] = value[i, :len_1[i]]

        # compute the number of mixed tokens in mask
        mix_1 = torch.sum(val_1 == 2, dim=1)

        # create intermediate list to hold vectors
        x_0 = torch.zeros((batch_size, torch.max(mix_1), self.embed_dim), device=value.device)

        # deconvolute the latent space - sequence length equals number of tokens in the penultimate layer
        y_1 = self.deconvolution_1(x)
        # select only latent vectors, which correspond to mixed tokens in the penultimate layer
        for i in range(batch_size):
            mix_1_mask_i = (val_1[i] == 2)
            x_0[i, :torch.sum(mix_1_mask_i)] = y_1[i, mix_1_mask_i]  # [N, T', C]

        # deconvolute the intermediate latent space - create new tokens in latent space for each mixed token
        y_0 = self.deconvolution_0(x_0)  # [N, T, C]

        # add spatial decoding if available
        if self.spatial_encoding is not None:
            len_last = torch.sum(depth == max_depth, dim=1)
            assert ((depth[:, -len_last:] == max_depth).all())
            y_0 = y_0 + self.spatial_encoding(pos[:, -len_last:])

        # compute logits of generated tokens
        return self.linear(y_0)  # [N, T, V]
