import math
import torch.nn as nn

from masks import padding_mask


class SingleConvolutionalEmbeddingA(nn.Module):
    def __init__(self, num_vocab, embed_dim, resolution, spatial_dim):
        """ Performs an embedding of token sequences into an embedding space of higher dimension.

        Uses a convolution to reduce the number of tokens with 's' as the kernel size and stride, where 's' is
        2^`spatial_dim`.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different token values (exclusive padding token '0').
            embded_dim: Dimension of returned embedding space.
            resolution: Spatial resolution of sequence encoding.
            spatial_dim: Spatial dimension (2D, 3D, ...) of sequence encoding.
        """
        super(SingleConvolutionalEmbeddingA, self).__init__()
        self.embed_dim = embed_dim
        self.spatial_dim = spatial_dim
        tree_depth = int(math.log2(resolution))

        # embeddings
        self.src_value_embedding = nn.Embedding(num_vocab + 1, embed_dim, padding_idx=0)
        self.tgt_value_embedding = nn.Embedding(num_vocab + 1, embed_dim, padding_idx=0)
        self.depth_embedding = nn.Embedding(tree_depth + 1, embed_dim, padding_idx=0)
        self.spatial_embeddings = nn.ModuleList(
            [nn.Embedding(2 * resolution, embed_dim, padding_idx=0) for _ in range(spatial_dim)]
        )

        # convolutions
        s = 2**spatial_dim
        self.src_convolution = nn.Conv1d(embed_dim, embed_dim, kernel_size=s, stride=s)
        self.tgt_convolution = nn.Conv1d(embed_dim, embed_dim, kernel_size=s, stride=s)

    def _embed_depth_position(self, depth, position):
        """ Adds embedding of depth and position into embedding space. """
        x = self.depth_embedding(depth)
        for axis, spatial_embedding in enumerate(self.spatial_embeddings):
            x = x + spatial_embedding(position[:, :, axis])
        return x

    def source(self, value, depth, position):
        """ Transform sequences into embedding space for the encoder and reduce number of tokens.

        Args:
            value: Value token sequence for the encoder input.
            depth: Depth token sequence for the encoder input.
            position: Position token sequence for the encoder input.

        Return:
            Reduced token sequence in the embedding space.
        """
        # embed tokens into higher dimension
        x = self.src_value_embedding(value) + self._embed_depth_position(depth, position)  # [N, S, E]
        # convolute tokens to reduce sequence length
        return self.src_convolution(x.transpose(1, 2)).transpose(1, 2)  # [N, S', E]

    def target(self, value, depth, position):
        """ Transform sequences into embedding space for the decoder and reduce number of tokens.

        Args:
            value: Value token sequence for the decoder input.
            depth: Depth token sequence for the decoder input.
            position: Position token sequence for the decoder input.

        Return:
            Reduced token sequence in the embedding space.
        """
        # embed tokens into higher dimension
        x = self.tgt_value_embedding(value) + self._embed_depth_position(depth, position)  # [N, T, E]
        # convolute tokens to reduce sequence length
        return self.tgt_convolution(x.transpose(1, 2)).transpose(1, 2)  # [N, T', E]

    def src_padding_mask(self, value, depth):
        """ Creates a token padding mask for the encoder, based on the value and depth sequence token.

        Uses only every n-th value token as input, where n is the convolution kernel size.

        Args:
            value: Value token sequence for the encoder input.
            depth: Depth token sequence for the encoder input.

        Return:
            Padding mask, where padding tokens '0' of the value sequence are masked out.
        """
        return padding_mask(value[:, ::2**self.spatial_dim], device=value.device)

    def tgt_padding_mask(self, value, depth):
        """ Creates a token padding mask for the decoder, based on the value and depth sequence token.

        Uses only every n-th value token as input, where n is the convolution kernel size.

        Args:
            value: Value token sequence for the decoder input.
            depth: Depth token sequence for the decoder input.

        Return:
            Padding mask, where padding tokens '0' of the value sequence are masked out.
        """
        return padding_mask(value[:, ::2**self.spatial_dim], device=value.device)
