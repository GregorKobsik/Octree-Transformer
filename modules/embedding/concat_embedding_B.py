import math
import torch
import torch.nn as nn

from masks import padding_mask


class ConcatEmbeddingB(nn.Module):
    def __init__(self, num_vocab, embed_dim, resolution, spatial_dim):
        """ Performs an embedding of token sequences into an embedding space of higher dimension.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different token values (exclusive padding token '0').
            embded_dim: Dimension of returned embedding space.
            resolution: Spatial resolution of sequence encoding.
            spatial_dim: Spatial dimension (2D, 3D, ...) of sequence encoding.
        """
        super(ConcatEmbeddingB, self).__init__()
        self.embed_dim = embed_dim
        self.spatial_dim = spatial_dim
        tree_depth = int(math.log2(resolution))
        self.chunk_size = 2**spatial_dim
        e_dim = embed_dim // self.chunk_size * 2

        # embeddings
        self.value_embedding = nn.Embedding(num_vocab + 1, e_dim, padding_idx=0)
        self.depth_embedding = nn.Embedding(tree_depth + 1, e_dim, padding_idx=0)
        self.spatial_embeddings = nn.ModuleList(
            [nn.Embedding(2 * resolution, e_dim, padding_idx=0) for _ in range(spatial_dim)]
        )

        # layers
        layers = []
        layers.append(nn.LayerNorm(self.chunk_size * e_dim))
        layers.append(nn.Linear(self.chunk_size * e_dim, embed_dim))
        layers.append(nn.GELU())
        self.feed_forward = nn.Sequential(*layers)

    def _embed_sequences(self, value, depth, position):
        """ Adds embedding of depth and position into embedding space. """
        x = self.value_embedding(value)
        x = x + self.depth_embedding(depth)
        for axis, spatial_embedding in enumerate(self.spatial_embeddings):
            x = x + spatial_embedding(position[:, :, axis])
        return x

    def _split_and_cat(self, x):
        """ Reshape the tensor to concatenate 2^`spatial_dim` tokens along the embedding dimension."""
        batch_size, num_tokens, embed_dim = x.shape
        x = torch.reshape(x, (batch_size, num_tokens // self.chunk_size, -1))
        return x

    def source(self, value, depth, position):
        """ Transform sequences into embedding space for the encoder.

        Args:
            value: Value token sequence for the encoder input.
            depth: Depth token sequence for the encoder input.
            position: Position token sequence for the encoder input.

        Return:
            Token sequence in the embedding space.
        """
        # embed the tokens into higher dimensional space
        x = self._embed_sequences(value, depth, position)  # [N, S, E']
        # concatenate the embedding dimension of each n-tokens to reduce their size
        x = self._split_and_cat(x)  # [N, S', E]
        # apply fully connected layer with nonlinearity
        x = self.feed_forward(x)

        return x

    def target(self, value, depth, position):
        """ Transform sequences into embedding space for the decoder.

        Args:
            value: Value token sequence for the decoder input.
            depth: Depth token sequence for the decoder input.
            position: Position token sequence for the decoder input.

        Return:
            Token sequence in the embedding space.
        """
        # embed the tokens into higher dimensional space
        x = self._embed_sequences(value, depth, position)  # [N, S, E']
        # concatenate the embedding dimension of each n-tokens to reduce their size
        x = self._split_and_cat(x)  # [N, S', E]
        # apply fully connected layer with nonlinearity
        x = self.feed_forward(x)

        return x

    def src_padding_mask(self, value, depth):
        """ Creates a token padding mask for the encoder, based on the value and depth sequence token.

        Uses only every n-th value token as input, where n is the convolution kernel size.

        Args:
            value: Value token sequence for the encoder input.
            depth: Depth token sequence for the encoder input.

        Return:
            Padding mask, where padding tokens '0' of the value sequence are masked out.
        """
        return padding_mask(value[:, ::self.chunk_size], device=value.device)

    def tgt_padding_mask(self, value, depth):
        """ Creates a token padding mask for the decoder, based on the value and depth sequence token.

        Uses only every n-th value token as input, where n is the convolution kernel size.

        Args:
            value: Value token sequence for the decoder input.
            depth: Depth token sequence for the decoder input.

        Return:
            Padding mask, where padding tokens '0' of the value sequence are masked out.
        """
        return padding_mask(value[:, ::self.chunk_size], device=value.device)
