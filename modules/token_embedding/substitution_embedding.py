import torch
import torch.nn as nn

from utils.masks import padding_mask
from ..utils import Convolution


class SubstitutionEmbedding(nn.Module):
    def __init__(self, encoding, num_vocab, embed_dim, resolution, spatial_dim, conv_size, **_):
        """ Performs an embedding of token sequences into an embedding space of higher dimension.

        The embedding packs multiple tokens of the last sequence into one token embedded in higher dimension using
        convolutional operations. The sequence is packed twice, therefore one token in the input is responsible for
        multiple tokens in the output and some tokens of the input might be discarded in the process.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            encoding: Defines how the tokens are encoded before being reduced
            num_vocab: Number of different token values (exclusive padding token '0').
            embded_dim: Dimension of returned embedding space.
            resolution: Spatial resolution of sequence encoding.
            spatial_dim: Spatial dimension (2D, 3D, ...) of sequence encoding.
            conv_size: Convolution kernel size and stride.
        """
        super(SubstitutionEmbedding, self).__init__()
        self.conv_size = conv_size
        self.spatial_dim = spatial_dim
        self.mask = None

        # embeddings
        self.embedding = encoding

        # convolutions
        self.convolution_0 = Convolution(embed_dim, embed_dim, 8)
        self.convolution_1 = Convolution(embed_dim, embed_dim, conv_size)

    def reduce(self, embedding, value, depth, position):
        """ Transform sequences into embedding space for the encoder.

        Uses a convolutional operation to pack multiple tokens of the last layer into one token in higher dimension.
        All tokens of the last layer are used for the embedding and none discarded. Uses a convolutional operation to
        pack one token of the penultimate layer into one token in higher dimension and substitutes mixed ('2') token
        embeddings with the corresponding embeddings of the last layer. Therefore some tokens of the penultimate layer
        might be discarded.

        Args:
            embedding: embedding sequence, with penultimate and last layer.
            value: Value token sequence, with penultimate and last layer.
            depth: Depth token sequence, with penultimate and last layer.
            position: Position token sequence, with penultimate and last layer.

        Return:
            Token sequence in the embedding space.
        """
        batch_size = value.shape[0]
        max_depth = torch.max(depth)
        len_0 = torch.sum(depth == max_depth, dim=1)
        len_1 = torch.sum(depth == (max_depth - 1), dim=1)

        assert (len_0 + len_1 == value.shape[1])

        # create intermediate list to hold values
        x_0 = torch.zeros((batch_size, torch.max(len_0), embedding.shape[2]), dtype=torch.float, device=value.device)
        val_0 = torch.zeros((batch_size, torch.max(len_0)), dtype=torch.long, device=value.device)
        dep_0 = torch.zeros((batch_size, torch.max(len_0)), dtype=torch.long, device=value.device)
        pos_0 = torch.zeros((batch_size, torch.max(len_0), self.spatial_dim), dtype=torch.long, device=value.device)

        x_1 = torch.zeros((batch_size, torch.max(len_1), embedding.shape[2]), dtype=torch.float, device=value.device)
        val_1 = torch.zeros((batch_size, torch.max(len_1)), dtype=torch.long, device=value.device)
        dep_1 = torch.zeros((batch_size, torch.max(len_1)), dtype=torch.long, device=value.device)
        pos_1 = torch.zeros((batch_size, torch.max(len_1), self.spatial_dim), dtype=torch.long, device=value.device)

        # split input in penultimate (1) and last (0) layer
        for i in range(batch_size):
            x_1[i, :len_1[i]] = embedding[i, :len_1[i]]
            val_1[i, :len_1[i]] = value[i, :len_1[i]]
            dep_1[i, :len_1[i]] = depth[i, :len_1[i]]
            pos_1[i, :len_1[i]] = position[i, :len_1[i]]
            x_0[i, :len_0[i]] = embedding[i, len_1[i]:len_1[i] + len_0[i]]
            val_0[i, :len_0[i]] = value[i, len_1[i]:len_1[i] + len_0[i]]
            dep_0[i, :len_0[i]] = depth[i, len_1[i]:len_1[i] + len_0[i]]
            pos_0[i, :len_0[i]] = position[i, len_1[i]:len_1[i] + len_0[i]]

        # precompute padding mask
        self.mask = padding_mask(val_1[:, ::self.conv_size], device=value.device)

        # convolve embedded tokens of last layer
        y_0 = self.convolution_0(x_0)  # [N, T2', C]

        # substitute all mixed token embeddings of penultimate layer, with token embeddings of last layer
        x_1[val_1 == 2] = y_0[val_0[:, ::8] != 0]  # [N, T1, C]

        # convolve substituted tokens of penultimate layer
        return self.convolution_1(x_1.contiguous())  # [N, T1', E]

    def forward(self, value, depth, position):
        """ Transform sequences into embedding space for the encoder.

        Uses a convolutional operation to pack multiple tokens of the last layer into one token in higher dimension.
        All tokens of the last layer are used for the embedding and none discarded. Uses a convolutional operation to
        pack one token of the penultimate layer into one token in higher dimension and substitutes mixed ('2') token
        embeddings with the corresponding embeddings of the last layer. Therefore some tokens of the penultimate layer
        might be discarded.

        Args:
            value: Value token sequence, with penultimate and last layer.
            depth: Depth token sequence, with penultimate and last layer.
            position: Position token sequence, with penultimate and last layer.

        Return:
            Token sequence in the embedding space.
        """

        return self.reduce(self.embedding(value, depth, position), value, depth, position)

    def padding_mask(self):
        """ Returns a padding mask, where padding tokens '0' of the value sequence are masked out. """
        return self.mask
