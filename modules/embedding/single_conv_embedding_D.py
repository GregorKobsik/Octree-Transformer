import math
import torch.nn as nn

from masks import padding_mask


class SingleConvolutionalEmbeddingD(nn.Module):
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
        super(SingleConvolutionalEmbeddingD, self).__init__()
        self.embed_dim = embed_dim
        self.spatial_dim = spatial_dim
        tree_depth = int(math.log2(resolution))

        # embeddings
        self.src_value_embedding = nn.Embedding(num_vocab + 1, embed_dim, padding_idx=0)
        self.depth_embedding = nn.Embedding(tree_depth + 1, embed_dim, padding_idx=0)
        self.spatial_embeddings = nn.ModuleList(
            [nn.Embedding(2 * resolution, embed_dim, padding_idx=0) for _ in range(spatial_dim)]
        )

        # convolutions
        s = 2**spatial_dim
        self.tgt_value_convolution = nn.Conv1d(1, embed_dim, kernel_size=s, stride=s)
        self.depth_convolution = nn.Conv1d(1, embed_dim, kernel_size=s, stride=s)
        self.spatial_convolutions = nn.ModuleList(
            [nn.Conv1d(1, embed_dim, kernel_size=s, stride=s) for _ in range(spatial_dim)]
        )

    def source(self, value, depth, position):
        """ Transform sequences into embedding space for the encoder.

        Args:
            value: Value token sequence for the encoder input.
            depth: Depth token sequence for the encoder input.
            position: Position token sequence for the encoder input.

        Return:
            Reduced token sequence in the embedding space.
        """
        # embed tokens into higher dimension
        x = self.src_value_embedding(value)  # [N, S, E]
        x = x + self.depth_embedding(depth)
        for axis, spatial_embedding in enumerate(self.spatial_embeddings):
            x = x + spatial_embedding(position[:, :, axis])
        return x  # [N, S, E]

    def target(self, value, depth, position):
        """ Transform sequences into embedding space for the decoder and reduce number of tokens.

        Args:
            value: Value token sequence for the decoder input.
            depth: Depth token sequence for the decoder input.
            position: Position token sequence for the decoder input.

        Return:
            Reduced token sequence in the embedding space.
        """
        # cast sequences to 'float', as convolutions do not support 'long'
        value = value.float()
        depth = depth.float()
        position = position.float()

        # embed tokens into higher dimension
        x = self.tgt_value_convolution(value.unsqueeze(1))  # [N, E, T']
        x = x + self.depth_convolution(depth.unsqueeze(1))  # [N, E, T']
        for axis, spatial_conv in enumerate(self.spatial_convolutions):
            x = x + spatial_conv(position[:, :, axis].unsqueeze(1))  # [N, E, T']
        # transpose dimensions
        return x.transpose(1, 2)  # [N, T', E]

    def src_padding_mask(self, value, depth):
        """ Creates a token padding mask for the encoder, based on the value and depth sequence token.

        Uses only every n-th value token as input, where n is the convolution kernel size.

        Args:
            value: Value token sequence for the encoder input.
            depth: Depth token sequence for the encoder input.

        Return:
            Padding mask, where padding tokens '0' of the value sequence are masked out.
        """
        return padding_mask(value, device=value.device)

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
