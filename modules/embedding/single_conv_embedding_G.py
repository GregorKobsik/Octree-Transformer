import math
import torch.nn as nn

from masks import padding_mask


class SingleConvolutionalEmbeddingG(nn.Module):
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
        super(SingleConvolutionalEmbeddingG, self).__init__()
        self.spatial_dim = spatial_dim
        tree_depth = int(math.log2(resolution))

        # convolutions
        self.conv_x = nn.Sequential(
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=2, stride=2),
            nn.GELU(),
        )
        embed_dim = embed_dim // 2
        self.conv_y = nn.Sequential(
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=2, stride=2),
            nn.GELU(),
        )
        embed_dim = embed_dim // 2
        if self.spatial_dim == 3:
            self.conv_z = nn.Sequential(
                nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=2, stride=2),
                nn.GELU(),
            )
            embed_dim = embed_dim // 2

        # embeddings
        self.value_embedding = nn.Embedding(num_vocab + 1, embed_dim, padding_idx=0)
        self.depth_embedding = nn.Embedding(tree_depth + 1, embed_dim, padding_idx=0)
        self.spatial_embeddings = nn.ModuleList(
            [nn.Embedding(2 * resolution, embed_dim, padding_idx=0) for _ in range(spatial_dim)]
        )

    def _embed(self, value, depth, position):
        """ Adds embedding of depth and position into embedding space. """
        x = self.value_embedding(value)
        x = x + self.depth_embedding(depth)
        for axis, spatial_embedding in enumerate(self.spatial_embeddings):
            x = x + spatial_embedding(position[:, :, axis])
        return x

    def _convolute(self, x):
        """ Convolutes two neighbouring sequence tokens into one in higher embedding space. """
        x = x.transpose(1, 2)
        if self.spatial_dim == 3:
            x = self.conv_z(x)
        x = self.conv_y(x)
        x = self.conv_x(x)
        return x.transpose(1, 2)

    def source(self, value, depth, position):
        """ Transform sequences into embedding space for the encoder and reduce number of tokens.

        Args:
            value: Value token sequence for the encoder input.
            depth: Depth token sequence for the encoder input.
            position: Position token sequence for the encoder input.

        Return:
            Reduced token sequence in the embedding space.
        """
        # embed tokens into embedding space
        x = self._embed(value, depth, position)  # [N, S, E']
        # convolute tokens to reduce sequence length
        return self._convolute(x)  # [N, S', E]

    def target(self, value, depth, position):
        """ Transform sequences into embedding space for the decoder and reduce number of tokens.

        Args:
            value: Value token sequence for the decoder input.
            depth: Depth token sequence for the decoder input.
            position: Position token sequence for the decoder input.

        Return:
            Reduced token sequence in the embedding space.
        """
        # embed tokens into embedding space
        x = self._embed(value, depth, position)  # [N, S, E']
        # convolute tokens to reduce sequence length
        return self._convolute(x)  # [N, S', E]

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
