import torch.nn as nn

from masks import padding_mask


class SingleConvolutionalEmbeddingB(nn.Module):
    def __init__(self, embed_dim, spatial_dim):
        """ Performs an embedding of token sequences into an embedding space of higher dimension.

        Uses a convolution to reduce the number of tokens with 's' as the kernel size and stride, where 's' is
        2^`spatial_dim`.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            embded_dim: Dimension of returned embedding space.
            spatial_dim: Spatial dimension (2D, 3D, ...) of sequence encoding.
        """
        super(SingleConvolutionalEmbeddingB, self).__init__()
        self.embed_dim = embed_dim
        self.spatial_dim = spatial_dim
        s = 2**spatial_dim

        # convolutions
        self.src_value_convolution = nn.Conv1d(1, embed_dim, kernel_size=s, stride=s)
        self.tgt_value_convolution = nn.Conv1d(1, embed_dim, kernel_size=s, stride=s)
        self.depth_convolution = nn.Conv1d(1, embed_dim, kernel_size=s, stride=s)
        self.spatial_convolution = nn.ModuleList(
            [nn.Conv1d(1, embed_dim, kernel_size=s, stride=s) for _ in range(spatial_dim)]
        )

    def _convolute_depth_position(self, depth, position):
        """ Adds embedding of depth and position into embedding space. """
        x = self.depth_convolution(depth.unsqueeze(1))
        for axis, spatial_conv in enumerate(self.spatial_convolution):
            x = x + spatial_conv(position[:, :, axis].unsqueeze(1))
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
        # cast sequences to 'float', as convolutions do not support 'long'
        value = value.float()
        depth = depth.float()
        position = position.float()

        # embed tokens into higher dimension
        x = self.src_value_convolution(value.unsqueeze(1))  # [N, E, S']
        x = x + self._convolute_depth_position(depth, position)  # [N, E, S']
        # convolute tokens to reduce sequence length
        return x.transpose(1, 2)  # [N, S', E]

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
        x = x + self._convolute_depth_position(depth, position)  # [N, E, T']
        # convolute tokens to reduce sequence length
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
