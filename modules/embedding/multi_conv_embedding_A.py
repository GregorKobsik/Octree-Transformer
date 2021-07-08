import math
import torch.nn as nn

from masks import padding_mask


class MultiConvolutionalEmbeddingA(nn.Module):
    def __init__(self, num_vocab, embed_dim, resolution, spatial_dim):
        """ Performs an embedding of token sequences into an embedding space of higher dimension.

        Uses a convolution to reduce the number of tokens with 's' as the kernel size and stride, where 's' is
        2^(`spatial_dim` - 1).

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different token values (exclusive padding token '0').
            embded_dim: Dimension of returned embedding space.
            resolution: Spatial resolution of sequence encoding.
            spatial_dim: Spatial dimension (2D, 3D, ...) of sequence encoding.
        """
        super(MultiConvolutionalEmbeddingA, self).__init__()
        tree_depth = int(math.log2(resolution))
        conv_dim = embed_dim // 2

        # embeddings
        self.value_embedding = nn.Embedding(num_vocab + 1, conv_dim, padding_idx=0)
        self.depth_embedding = nn.Embedding(tree_depth + 1, conv_dim, padding_idx=0)
        self.spatial_embeddings = nn.ModuleList(
            [nn.Embedding(2 * resolution, conv_dim, padding_idx=0) for _ in range(spatial_dim)]
        )

        # convolutions
        self.chunk_size = 2**(spatial_dim)
        self.convolution = nn.Conv1d(conv_dim, conv_dim, kernel_size=self.chunk_size, stride=self.chunk_size)
        self.multi_convolution = nn.Conv1d(conv_dim, embed_dim, kernel_size=self.chunk_size, stride=self.chunk_size)

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
        x = self.value_embedding(value)  # [N, S, C]
        x = x + self.depth_embedding(depth)  # [N, S, C]
        for axis, spatial_embedding in enumerate(self.spatial_embeddings):
            x = x + spatial_embedding(position[:, :, axis])  # [N, S, C]

        # convolute tokens to reduce sequence length
        y = self.convolution(x.transpose(1, 2)).transpose(1, 2)  # [N, S', C]

        # convolute tokens to reduce sequence length even more
        return self.multi_convolution(y.transpose(1, 2)).transpose(1, 2)  # [N, S'', E]

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
