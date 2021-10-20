import torch.nn as nn

from utils.masks import padding_mask
from ..utils import Convolution


class ConvolutionEmbeddingA(nn.Module):
    def __init__(self, encoding, num_vocab, embed_dim, resolution, spatial_dim, conv_size, **_):
        """ Performs an embedding of token sequences into an embedding space of higher dimension.

        Uses a convolution to reduce the number of tokens with 's' as the kernel size and stride, where 's' is
        2^(`spatial_dim`).

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            encoding: Defines how the tokens are encoded before being reduced
            num_vocab: Number of different token values (exclusive padding token '0').
            embded_dim: Dimension of returned embedding space.
            resolution: Spatial resolution of sequence encoding.
            spatial_dim: Spatial dimension (2D, 3D, ...) of sequence encoding.
            conv_size: Convolution kernel size and stride.
        """
        super(ConvolutionEmbeddingA, self).__init__()
        self.chunk_size = conv_size

        # embeddings
        self.embedding = encoding
        self.convolution = Convolution(embed_dim, embed_dim, conv_size)

    def reduce(self, embedding, value, depth, position):
        """ Transform sequences of token into an embedding space and reduces number of tokens.

        Args:
            value: Embedding sequence.
            value: Value token sequence.
            depth: Depth token sequence.
            position: Position token sequence.

        Return:
            Token sequence in the embedding space.
        """
        # convolute tokens to reduce sequence length
        return self.convolution(embedding)  # [N, S', E]

    def forward(self, value, depth, position):
        """ Transform sequences of token into an embedding space and reduces number of tokens.

        Args:
            value: Value token sequence.
            depth: Depth token sequence.
            position: Position token sequence.

        Return:
            Token sequence in the embedding space.
        """
        return self.reduce(self.embedding(value, depth, position), value, depth, position)

    def padding_mask(self, value, depth, position):
        """ Creates a token padding mask, based on the value and depth sequence token.

        Uses only every n-th value token as input, where n is the convolution kernel size.

        Args:
            value: Value token sequence.
            depth: Depth token sequence.
            position: Position token sequence.

        Return:
            Padding mask, where padding tokens '0' of the value sequence are masked out.
        """
        return padding_mask(value[:, ::self.chunk_size], device=value.device)
