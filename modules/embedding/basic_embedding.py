import math
import torch.nn as nn

from masks import padding_mask


class BasicEmbedding(nn.Module):
    def __init__(self, num_vocab, embed_dim, resolution, spatial_dim):
        """ Performs an embedding of token sequences into an embedding space of higher dimension.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different token values (exclusive padding token '0').
            embded_dim: Dimension of returned embedding space.
            resolution: Spatial resolution of sequence encoding.
            spatial_dim: Spatial dimension (2D, 3D, ...) of sequence encoding.
        """
        super(BasicEmbedding, self).__init__()
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

    def _embed_depth_position(self, depth, position):
        """ Adds embedding of depth and position into embedding space. """
        x = self.depth_embedding(depth)
        for axis, spatial_embedding in enumerate(self.spatial_embeddings):
            x = x + spatial_embedding(position[:, :, axis])
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
        x = self.src_value_embedding(value)
        return x + self._embed_depth_position(depth, position)

    def target(self, value, depth, position):
        """ Transform sequences into embedding space for the decoder.

        Args:
            value: Value token sequence for the decoder input.
            depth: Depth token sequence for the decoder input.
            position: Position token sequence for the decoder input.

        Return:
            Token sequence in the embedding space.
        """
        x = self.tgt_value_embedding(value)
        return x + self._embed_depth_position(depth, position)

    def src_padding_mask(self, value, depth):
        """ Creates a token padding mask for the encoder, based on the value and depth sequence token.

        Args:
            value: Value token sequence for the encoder input.
            depth: Depth token sequence for the encoder input.

        Return:
            Padding mask, where padding tokens '0' of the value sequence are masked out.
        """
        return padding_mask(value, device=value.device)

    def tgt_padding_mask(self, value, depth):
        """ Creates a token padding mask for the decoder, based on the value and depth sequence token.

        Args:
            value: Value token sequence for the decoder input.
            depth: Depth token sequence for the decoder input.

        Return:
            Padding mask, where padding tokens '0' of the value sequence are masked out.
        """
        return padding_mask(value, device=value.device)
