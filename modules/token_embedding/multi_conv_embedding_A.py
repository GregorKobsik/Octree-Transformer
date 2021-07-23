import torch.nn as nn

from modules.utils import Embedding, Convolution
from utils.masks import padding_mask


class MultiConvolutionEmbeddingA(nn.Module):
    def __init__(self, num_vocab, embed_dim, resolution, spatial_dim, conv_size, **_):
        """ Performs an embedding of token sequences into an embedding space of higher dimension.

        Uses a convolution to reduce the number of tokens with 's' as the kernel size and stride, where 's' is
        2^(`spatial_dim` - 1).

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different token values (exclusive padding token '0').
            embded_dim: Dimension of returned embedding space.
            resolution: Spatial resolution of sequence encoding.
            spatial_dim: Spatial dimension (2D, 3D, ...) of sequence encoding.
            conv_size: Convolution kernel size and stride.
        """
        super(MultiConvolutionEmbeddingA, self).__init__()
        self.chunk_size = conv_size

        # embeddings
        self.embedding = Embedding(embed_dim // 4, num_vocab, resolution, spatial_dim)
        self.convolution_1 = Convolution(embed_dim // 4, embed_dim // 2, conv_size)
        self.convolution_2 = Convolution(embed_dim // 2, embed_dim, conv_size)

    def forward(self, value, depth, position):
        """ Transform sequences into embedding space for the encoder and reduce number of tokens.

        Args:
            value: Value token sequence.
            depth: Depth token sequence.
            position: Position token sequence.

        Return:
            Reduced token sequence in the embedding space.
        """
        # embed tokens into higher dimension
        x = self.embedding(value, depth, position)  # [N, S, C]

        # convolute tokens to reduce sequence length
        y = self.convolution_1(x)  # [N, S', C]

        # convolute tokens to reduce sequence length even more
        return self.convolution_2(y)  # [N, S'', E]

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
        return padding_mask(value[:, ::self.chunk_size**2], device=value.device)
