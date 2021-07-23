import math
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, embed_dim, num_vocab, resolution, spatial_dim):
        """ Performs an embedding of token sequences into an embedding space of higher dimension.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            embded_dim: Dimension of returned embedding space.
            num_vocab: Number of different token values (exclusive padding token '0').
            resolution: Spatial resolution of sequence encoding.
            spatial_dim: Spatial dimension (2D, 3D, ...) of sequence encoding.
        """
        super(Embedding, self).__init__()
        tree_depth = int(math.log2(resolution))

        # embeddings
        self.value_embedding = nn.Embedding(num_vocab + 1, embed_dim, padding_idx=0)
        self.depth_embedding = nn.Embedding(tree_depth + 1, embed_dim, padding_idx=0)
        self.spatial_embeddings = nn.ModuleList(
            [nn.Embedding(2 * resolution, embed_dim, padding_idx=0) for _ in range(spatial_dim)]
        )

    def forward(self, value, depth, position):
        """ Transform sequences of token into an embedding space.

        Args:
            value: Value token sequence with the shape [N, S].
            depth: Depth token sequence with the shape [N, S].
            position: Position token sequence with the shape [N, S, A].

        Return:
            Token sequence in the embedding space with the shape [N, S, E].
        """
        x = self.value_embedding(value)  # [N, S, E]
        if depth is not None:
            x = x + self.depth_embedding(depth)  # [N, S, E]
        for axis, spatial_embedding in enumerate(self.spatial_embeddings):
            x = x + spatial_embedding(position[:, :, axis])  # [N, S, E]
        return x  # [N, S, E]
