import torch
import torch.nn as nn


class PositionalEncodingLearned(nn.Module):
    def __init__(self, embed_dim, resolution):
        """ Performs an embedding of token sequences into an embedding space of higher dimension.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            embed_dim: Dimension of returned embedding space.
            resolution: Spatial resolution of sequence encoding.
        """
        super(PositionalEncodingLearned, self).__init__()

        self.embed_dim = embed_dim
        self.spatial_embeddings = nn.ModuleList(
            [nn.Embedding(2 * resolution, embed_dim, padding_idx=0) for _ in range(3)]
        )

    def forward(self, position):
        """ Transform sequences of token into an embedding space.

        Args:
            position: Position token sequence with the shape [N, S, A].

        Return:
            Token sequence in the embedding space with the shape [N, S, E].
        """
        x = torch.zeros((position.shape[0], position.shape[1], self.embed_dim), device=position.device)
        for axis, spatial_embedding in enumerate(self.spatial_embeddings):
            x = x + spatial_embedding(position[:, :, axis])  # [N, S, E]
        return x  # [N, S, E]


class PositionalEncodingLearnedLookAhead(nn.Module):
    def __init__(self, embed_dim, resolution):
        """ Performs an embedding of token sequences into an embedding space of higher dimension.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            embed_dim: Dimension of returned embedding space.
            resolution: Spatial resolution of sequence encoding.
        """
        super(PositionalEncodingLearnedLookAhead, self).__init__()

        self.embed_dim = embed_dim
        self.spatial_embeddings = nn.ModuleList(
            [nn.Embedding(2 * resolution, embed_dim, padding_idx=0) for _ in range(3)]
        )
        # end of sequence positional token
        self.eos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.eos)

    def _append_eos_token(self, x):
        """ Appends eos token to sequence """
        batch_size = x.shape[0]
        eos = torch.ones(batch_size, 1, self.eos.shape[0], device=x.device) * self.eos  # [N, 1, E]
        return torch.cat([x, eos], dim=1)  # [N, S, E]

    def forward(self, position):
        """ Transform sequences of token into an embedding space.

        Args:
            position: Position token sequence with the shape [N, S, A].

        Return:
            Token sequence in the embedding space with the shape [N, S, E].
        """
        x = torch.zeros((position.shape[0], position.shape[1], self.embed_dim), device=position.device)
        for axis, spatial_embedding in enumerate(self.spatial_embeddings):
            x = x + spatial_embedding(position[:, :, axis])  # [N, S, E]

        x = self._append_eos_token(x)
        x = x[:, :-1] + x[:, 1:]

        return x  # [N, S, E]


class PositionalEncodingLearnedLookAheadSplit(nn.Module):
    def __init__(self, embed_dim, resolution):
        """ Performs an embedding of token sequences into an embedding space of higher dimension.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            embed_dim: Dimension of returned embedding space.
            resolution: Spatial resolution of sequence encoding.
        """
        super(PositionalEncodingLearnedLookAheadSplit, self).__init__()

        self.embed_dim = embed_dim
        self.spatial_embeddings = nn.ModuleList(
            [nn.Embedding(2 * resolution, embed_dim, padding_idx=0) for _ in range(3)]
        )
        self.spatial_embeddings_look_ahead = nn.ModuleList(
            [nn.Embedding(2 * resolution, embed_dim, padding_idx=0) for _ in range(3)]
        )
        # end of sequence positional token
        self.eos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.eos)

    def _append_eos_token(self, x):
        """ Appends eos token to sequence """
        batch_size = x.shape[0]
        eos = torch.ones(batch_size, 1, self.eos.shape[0], device=x.device) * self.eos  # [N, 1, E]
        return torch.cat([x, eos], dim=1)  # [N, S, E]

    def forward(self, position):
        """ Transform sequences of token into an embedding space.

        Args:
            position: Position token sequence with the shape [N, S, A].

        Return:
            Token sequence in the embedding space with the shape [N, S, E].
        """
        x = torch.zeros((position.shape[0], position.shape[1], self.embed_dim), device=position.device)
        for axis, spatial_embedding in enumerate(self.spatial_embeddings):
            x = x + spatial_embedding(position[:, :, axis])  # [N, S, E]

        x_look_ahead = torch.zeros_like(x)
        for axis, spatial_embedding in enumerate(self.spatial_embeddings_look_ahead):
            x_look_ahead = x_look_ahead + spatial_embedding(position[:, :, axis])  # [N, S, E]

        x_look_ahead = self._append_eos_token(x_look_ahead)
        x = x + x_look_ahead[:, 1:]

        return x  # [N, S, E]


class Embedding(nn.Module):
    def __init__(self, spatial_embedding, embed_dim, num_vocab):
        """ Performs an embedding of token sequences into an embedding space of higher dimension.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            spatial_embedding: A module that computes a spatiaal embedding based on the position
            embed_dim: Dimension of returned embedding space.
            num_vocab: Number of different token values (exclusive padding token '0').
        """
        super(Embedding, self).__init__()

        # embeddings
        self.value_embedding = nn.Embedding(num_vocab + 1, embed_dim, padding_idx=0)
        self.spatial_embedding = spatial_embedding

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
        x += self.spatial_embedding(position)
        return x  # [N, S, E]
