import math
import torch.nn as nn

from masks import padding_mask


class SingleConvolutionalEmbeddingH(nn.Module):
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
        super(SingleConvolutionalEmbeddingH, self).__init__()
        self.spatial_dim = spatial_dim
        tree_depth = int(math.log2(resolution))
        self.num_chunks = 2**spatial_dim
        e_dim = embed_dim // self.num_chunks
        layers = []

        # value embedding/convolutions
        self.value_embedding = nn.Sequential(nn.Embedding(num_vocab + 1, e_dim, padding_idx=0))
        for _ in range(spatial_dim):
            layers.append(nn.Conv1d(e_dim, 2 * e_dim, kernel_size=2, stride=2))
            layers.append(nn.GELU())
            e_dim = 2 * e_dim
        self.value_convolution = nn.Sequential(*layers)

        # depth & position embeddings
        self.depth_embedding = nn.Embedding(tree_depth + 1, e_dim, padding_idx=0)
        self.spatial_embeddings = nn.ModuleList(
            [nn.Embedding(2 * resolution, e_dim, padding_idx=0) for _ in range(spatial_dim)]
        )

        assert e_dim == embed_dim, "ERROR: Embedding dimension not divisable by 2^`spatial_dim`."

    def _embed_depth_position(self, depth, position):
        """ Adds embedding of depth and position into embedding space. """
        x = self.depth_embedding(depth)
        for axis, spatial_embedding in enumerate(self.spatial_embeddings):
            x = x + spatial_embedding(position[:, :, axis])
        return x

    def _convolute_value(self, x):
        """ Convolutes two neighbouring sequence tokens into one in higher embedding space. """
        x = x.transpose(1, 2)
        x = self.value_convolution(x)
        return x.transpose(1, 2)

    def _embed_sequences(self, value, depth, position):
        """ Embed given sequences into embedding space and reduce the number of tokens. """
        # embed value tokens into embedding space
        x = self.value_embedding(value)  # [N, S, E']

        # convolute tokens to reduce sequence length
        x = self._convolute_value(x)  # [N, S', E]

        # extract depth and position values of previous layer
        dep = depth[:, ::self.num_chunks]
        pos = (position[:, ::self.num_chunks] + position[:, (self.num_chunks - 1)::self.num_chunks]) // 2

        # add positional and depth embedding
        x = x + self._embed_depth_position(dep, pos)  # [N, S', E]

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
        return self._embed_sequences(value, depth, position)

    def target(self, value, depth, position):
        """ Transform sequences into embedding space for the decoder and reduce number of tokens.

        Args:
            value: Value token sequence for the decoder input.
            depth: Depth token sequence for the decoder input.
            position: Position token sequence for the decoder input.

        Return:
            Reduced token sequence in the embedding space.
        """
        return self._embed_sequences(value, depth, position)

    def src_padding_mask(self, value, depth):
        """ Creates a token padding mask for the encoder, based on the value and depth sequence token.

        Uses only every n-th value token as input, where n is the convolution kernel size.

        Args:
            value: Value token sequence for the encoder input.
            depth: Depth token sequence for the encoder input.

        Return:
            Padding mask, where padding tokens '0' of the value sequence are masked out.
        """
        return padding_mask(value[:, ::self.num_chunks], device=value.device)

    def tgt_padding_mask(self, value, depth):
        """ Creates a token padding mask for the decoder, based on the value and depth sequence token.

        Uses only every n-th value token as input, where n is the convolution kernel size.

        Args:
            value: Value token sequence for the decoder input.
            depth: Depth token sequence for the decoder input.

        Return:
            Padding mask, where padding tokens '0' of the value sequence are masked out.
        """
        return padding_mask(value[:, ::self.num_chunks], device=value.device)
