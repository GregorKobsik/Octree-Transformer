import torch.nn as nn

from ..utils import Deconvolution, Convolution, BlockConvolution, Linear


class ConvolutionHeadA(nn.Module):
    def __init__(self, spatial_encoding, num_vocab, embed_dim, head_dim, n_layer, conv_size, **_):
        """ Performs a convolutional transformation from transformer latent space into target value logits.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different target token values (exclusive padding token '0').
            embded_dim: Dimension of the latent embedding space of the transformer.
            head_dim: Size of embedding dimensions used in the head layers.
            n_layer: Number of layers used in each linear or convolution block.
            spatial_dim: Spatial dimension (2D/3D) of the sequence data.
            conv_size: Convolution kernel size and stride.
        """
        super(ConvolutionHeadA, self).__init__()

        deconvolution = [nn.GELU(), Deconvolution(embed_dim, head_dim, conv_size)]
        for i in range(n_layer - 1):
            deconvolution += [nn.GELU(), Convolution(head_dim, head_dim, (1,))]
        self.deconvolution = nn.Sequential(*deconvolution)

        linear = []
        for i in range(n_layer - 1):
            linear += [nn.GELU(), nn.Linear(head_dim, head_dim)]
        linear += [nn.GELU(), Linear(head_dim, num_vocab)]
        self.linear = nn.Sequential(*linear)

        self.spatial_encoding = spatial_encoding

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

        # add spatial decoding if available
        if self.spatial_encoding is not None:
            x = x + self.spatial_encoding(pos)

        # compute logits for each token
        return self.linear(x)  # [N, T, V]


class ConvolutionHeadAutoregressive(nn.Module):
    def __init__(self, spatial_encoding, num_vocab, embed_dim, head_dim, n_layer, conv_size, **_):
        """ Performs a convolutional transformation from transformer latent space into target value logits.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different target token values (exclusive padding token '0').
            embded_dim: Dimension of the latent embedding space of the transformer.
            head_dim: Size of embedding dimensions used in the head layers.
            n_layer: Number of layers used in each linear or convolution block.
            spatial_dim: Spatial dimension (2D/3D) of the sequence data.
            conv_size: Convolution kernel size and stride.
        """
        super(ConvolutionHeadAutoregressive, self).__init__()

        self.conv_size = conv_size

        deconvolution = [nn.GELU(), Deconvolution(embed_dim, head_dim, conv_size)]
        for i in range(n_layer - 1):
            deconvolution += [nn.GELU(), Convolution(head_dim, head_dim, (1,))]
        self.deconvolution = nn.Sequential(*deconvolution)

        convolution = [BlockConvolution(head_dim, head_dim, conv_size)]
        for i in range(n_layer - 1):
            convolution += [nn.GELU(), BlockConvolution(head_dim, head_dim, conv_size)]
        self.convolution = nn.Sequential(*convolution)

        linear = []
        for i in range(n_layer - 1):
            linear += [nn.GELU(), nn.Linear(head_dim, head_dim)]
        linear += [nn.GELU(), Linear(head_dim, num_vocab)]
        self.linear = nn.Sequential(*linear)

        self.spatial_encoding = spatial_encoding
        self.value_embedding = nn.Embedding(num_vocab + 1, head_dim, padding_idx=0)

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

        emb = self.value_embedding(value)
        # add spatial decoding if available
        if self.spatial_encoding is not None:
            emb = emb + self.spatial_encoding(pos)
        emb = self.convolution(emb[:, :x.shape[1]])

        x = x + emb

        # compute logits for each token
        return self.linear(x)  # [N, T, V]
