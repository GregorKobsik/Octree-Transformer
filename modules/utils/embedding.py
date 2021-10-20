import math

import torch
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


class LookAheadEmbedding(nn.Module):
    def __init__(self, embed_dim, num_vocab, resolution, spatial_dim):
        """ Performs an embedding of token sequences into an embedding space of higher dimension.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            embded_dim: Dimension of returned embedding space.
            num_vocab: Number of different token values (exclusive padding token '0').
            resolution: Spatial resolution of sequence encoding.
            spatial_dim: Spatial dimension (2D, 3D, ...) of sequence encoding.
        """
        super(LookAheadEmbedding, self).__init__()

        # embeddings
        self.value_embedding = nn.Embedding(num_vocab + 1, embed_dim, padding_idx=0)
        self.spatial_embeddings = nn.ModuleList(
            [nn.Embedding(2 * resolution, embed_dim, padding_idx=0) for _ in range(spatial_dim)]
        )
        # self.look_ahead = nn.Conv1d(embed_dim, embed_dim, 2)

        # end of sequence positional token
        self.eos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.eos)

    def _append_eos_token(self, x):
        """ Appends eos token to sequence """
        batch_size = x.shape[0]
        eos = torch.ones(batch_size, 1, self.eos.shape[0], device=x.device) * self.eos  # [N, 1, E]
        return torch.cat([x, eos], dim=1)  # [N, S, E]

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

        pos_embed = torch.zeros_like(x)
        for axis, spatial_embedding in enumerate(self.spatial_embeddings):
            pos_embed = pos_embed + spatial_embedding(position[:, :, axis])  # [N, S, E]

        pos_embed = self._append_eos_token(pos_embed)
        pos_embed = pos_embed[:, :-1] + pos_embed[:, 1:]
        x += pos_embed

        return x  # [N, S, E]


class LookAheadEmbeddingSplit(nn.Module):
    def __init__(self, embed_dim, num_vocab, resolution, spatial_dim):
        """ Performs an embedding of token sequences into an embedding space of higher dimension.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            embded_dim: Dimension of returned embedding space.
            num_vocab: Number of different token values (exclusive padding token '0').
            resolution: Spatial resolution of sequence encoding.
            spatial_dim: Spatial dimension (2D, 3D, ...) of sequence encoding.
        """
        super(LookAheadEmbeddingSplit, self).__init__()

        # embeddings
        self.value_embedding = nn.Embedding(num_vocab + 1, embed_dim, padding_idx=0)
        self.spatial_embeddings = nn.ModuleList(
            [nn.Embedding(2 * resolution, embed_dim, padding_idx=0) for _ in range(spatial_dim)]
        )
        self.spatial_embeddings_look_ahead = nn.ModuleList(
            [nn.Embedding(2 * resolution, embed_dim, padding_idx=0) for _ in range(spatial_dim)]
        )
        # self.look_ahead = nn.Conv1d(embed_dim, embed_dim, 2)

        # end of sequence positional token
        self.eos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.eos)

    def _append_eos_token(self, x):
        """ Appends eos token to sequence """
        batch_size = x.shape[0]
        eos = torch.ones(batch_size, 1, self.eos.shape[0], device=x.device) * self.eos  # [N, 1, E]
        return torch.cat([x, eos], dim=1)  # [N, S, E]

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

        pos_embed = torch.zeros_like(x)
        for axis, spatial_embedding in enumerate(self.spatial_embeddings):
            pos_embed = pos_embed + spatial_embedding(position[:, :, axis])  # [N, S, E]

        pos_embed_look_ahead = torch.zeros_like(x)
        for axis, spatial_embedding in enumerate(self.spatial_embeddings_look_ahead):
            pos_embed_look_ahead = pos_embed_look_ahead + spatial_embedding(position[:, :, axis])  # [N, S, E]

        pos_embed_look_ahead = self._append_eos_token(pos_embed)
        pos_embed = pos_embed + pos_embed_look_ahead[:, 1:]
        x += pos_embed

        return x  # [N, S, E]
