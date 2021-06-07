import math
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence


class DoubleConvolutionalHead(nn.Module):
    def __init__(self, num_vocab, embed_dim, spatial_dim):
        """ Performs a concolutional transformation from transformer latent space into target value logits.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different target token values (exclusive padding token '0').
            embded_dim: Dimension of the latent embedding space of the transformer.
            spatial_dim: Spatial dimension (2D/3D) of the sequence data.
        """
        super(DoubleConvolutionalHead, self).__init__()
        s = 2**spatial_dim
        conv_depth = math.sqrt(embed_dim)
        assert conv_depth == round(conv_depth), "The square root of `embed_dim` should be an integer."
        conv_depth = int(conv_depth)

        self.deconv_1 = nn.ConvTranspose1d(embed_dim, conv_depth, kernel_size=s, stride=s)
        self.deconv_2 = nn.ConvTranspose1d(conv_depth, 1, kernel_size=s, stride=s)
        self.linear = nn.Linear(1, num_vocab + 1, bias=False)

    def forward(self, x, value, depth, pos):
        """ Transforms the output of the transformer target value logits.

        Transforms one token of the latent vector into multiple tokens of the target vector through de-convolutional
        operations. In the case of a quadtree one token is responsible for up to 16 target tokens. In the case of a
        octree one token is responsible for up to 64 target tokens. Only tokens, which correspond to a mixed target
        value token in the penultimate layer are transformed into target sequence tokens.

        Args:
            x: Output of the transformer, the latent vector [N, T', E].
            value: Target value token sequence [N, T].
            depth: Target depth token sequence [N, T].
            pos: Target position token sequence [N, T, A].

        Return
            Logits of target value sequence.
        """
        # initialize variables
        batch_size = value.shape[0]
        val = batch_size * [0]
        y = batch_size * [0]

        # discard last layer of input sequence
        idx = torch.argmax(depth, dim=1)
        for i in range(batch_size):
            val[i] = value[i, :idx[i]]

        # repad sequences to equal length
        val_pad = pad_sequence(val, batch_first=True, padding_value=0)

        # deconvolute the latent space - resulting sequence length equals number of tokens in the penultimate layer
        x = x.transpose(1, 2).contiguous()  # [N, E, T'']
        x = self.deconv_1(x)  # [N, C, T']
        x = x.transpose(1, 2)  # [N, T', C]

        # filter out all values, which were not mixed in the penultimate layer
        for i in range(batch_size):
            y[i] = x[i, val_pad[i] == 2]
        y = pad_sequence(y, batch_first=True, padding_value=0)  # [N, T, C]

        # deconvolute the intermediate latent space - create new tokens in latent space for each mixed token
        y = y.transpose(1, 2).contiguous()  # [N, C, T]
        y = self.deconv_2(y)  # [N, 1, T]
        y = y.transpose(1, 2).contiguous()  # [N, T, 1]

        # compute logits of generated tokens
        return self.linear(y)  # [N, T, V]
