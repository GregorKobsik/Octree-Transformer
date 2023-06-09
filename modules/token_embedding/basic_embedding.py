import torch.nn as nn

from utils.masks import padding_mask


class BasicEmbedding(nn.Module):
    def __init__(self, encoding, num_vocab, embed_dim, resolution, spatial_dim, **_):
        """ Performs an embedding of token sequences into an embedding space of higher dimension.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            encoding: Defines how the tokens are encoded before being reduced
            num_vocab: Number of different token values (exclusive padding token '0').
            embded_dim: Dimension of returned embedding space.
            resolution: Spatial resolution of sequence encoding.
            spatial_dim: Spatial dimension (2D, 3D, ...) of sequence encoding.
        """
        super(BasicEmbedding, self).__init__()
        self.mask = None

        # embeddings
        self.embedding = encoding

    def reduce(self, embedding, value, depth, position):
        """ Transform sequences of token into an embedding space.

        Args:
            embedding: Embedding sequence
            value: Value token sequence.
            depth: Depth token sequence.
            position: Position token sequence.

        Return:
            Token sequence in the embedding space.
        """
        # precompute padding mask
        self.mask = padding_mask(value, device=value.device)

        return embedding

    def forward(self, value, depth, position):
        """ Transform sequences of token into an embedding space.

        Args:
            value: Value token sequence.
            depth: Depth token sequence.
            position: Position token sequence.

        Return:
            Token sequence in the embedding space.
        """
        return self.reduce(self.embedding(value, depth, position), value, depth, position)

    def padding_mask(self):
        """ Returns a padding mask, where padding tokens '0' of the value sequence are masked out. """
        return self.mask
