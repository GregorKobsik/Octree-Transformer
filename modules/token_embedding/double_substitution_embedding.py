import torch
import torch.nn as nn

from ..utils import Embedding, Convolution
from utils.masks import padding_mask


class DoubleSubstitutionEmbedding(nn.Module):
    def __init__(self, num_vocab, embed_dim, resolution, spatial_dim, conv_size, **_):
        """ Performs a double substitution embedding of token sequences into an embedding space of higher dimension.

        The embedding packs multiple tokens of the last sequence into one token embedded in higher dimension using
        convolutional operations. The sequence is packed twice, therefore one token in the input is responsible for
        multiple tokens in the output and some tokens of the input might be discarded in the process.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different token values (exclusive padding token '0').
            embded_dim: Dimension of returned embedding space.
            resolution: Spatial resolution of sequence encoding.
            spatial_dim: Spatial dimension (2D, 3D, ...) of sequence encoding.
            conv_size: Convolution kernel size and stride.
        """
        super(DoubleSubstitutionEmbedding, self).__init__()
        self.conv_size = conv_size
        self.spatial_dim = spatial_dim
        self.mask = None

        # embeddings
        self.embedding_0 = Embedding(embed_dim // 8, num_vocab, resolution, spatial_dim)
        self.embedding_1 = Embedding(embed_dim // 4, num_vocab, resolution, spatial_dim)
        self.embedding_2 = Embedding(embed_dim // 2, num_vocab, resolution, spatial_dim)

        # convolutions
        self.convolution_0 = Convolution(embed_dim // 8, embed_dim // 4, conv_size)
        self.convolution_1 = Convolution(embed_dim // 4, embed_dim // 2, conv_size)
        self.convolution_2 = Convolution(embed_dim // 2, embed_dim, conv_size)

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
        batch_size = value.shape[0]
        max_depth = torch.max(depth)

        # compute number of tokens in each layer
        len_0 = torch.sum(depth == max_depth, dim=1)
        len_1 = torch.sum(depth == (max_depth - 1), dim=1)
        len_2 = torch.sum(depth == (max_depth - 2), dim=1)

        # create intermediate list to hold values
        val_0 = torch.zeros((batch_size, torch.max(len_0)), dtype=torch.long, device=value.device)
        dep_0 = torch.zeros((batch_size, torch.max(len_0)), dtype=torch.long, device=value.device)
        pos_0 = torch.zeros((batch_size, torch.max(len_0), self.spatial_dim), dtype=torch.long, device=value.device)

        val_1 = torch.zeros((batch_size, torch.max(len_1)), dtype=torch.long, device=value.device)
        dep_1 = torch.zeros((batch_size, torch.max(len_1)), dtype=torch.long, device=value.device)
        pos_1 = torch.zeros((batch_size, torch.max(len_1), self.spatial_dim), dtype=torch.long, device=value.device)

        val_2 = torch.zeros((batch_size, torch.max(len_2)), dtype=torch.long, device=value.device)
        dep_2 = torch.zeros((batch_size, torch.max(len_2)), dtype=torch.long, device=value.device)
        pos_2 = torch.zeros((batch_size, torch.max(len_2), self.spatial_dim), dtype=torch.long, device=value.device)

        # splitt input in third-last (2), second-last (1) and last (0) layer
        for i in range(batch_size):
            val_2[i, :len_2[i]] = value[i, :len_2[i]]
            dep_2[i, :len_2[i]] = depth[i, :len_2[i]]
            pos_2[i, :len_2[i]] = position[i, :len_2[i]]

            val_1[i, :len_1[i]] = value[i, len_2[i]:len_2[i] + len_1[i]]
            dep_1[i, :len_1[i]] = depth[i, len_2[i]:len_2[i] + len_1[i]]
            pos_1[i, :len_1[i]] = position[i, len_2[i]:len_2[i] + len_1[i]]

            val_0[i, :len_0[i]] = value[i, len_2[i] + len_1[i]:len_2[i] + len_1[i] + len_0[i]]
            dep_0[i, :len_0[i]] = depth[i, len_2[i] + len_1[i]:len_2[i] + len_1[i] + len_0[i]]
            pos_0[i, :len_0[i]] = position[i, len_2[i] + len_1[i]:len_2[i] + len_1[i] + len_0[i]]

        # precompute padding mask
        self.mask = padding_mask(val_2[:, ::self.conv_size], device=value.device)  # [N, S'_2, E]

        # compute embeddings
        x_0 = self.embedding_0(val_0, dep_0, pos_0)  # [N, S_0, E // 8]
        x_1 = self.embedding_1(val_1, dep_1, pos_1)  # [N, S_1, E // 4]
        x_2 = self.embedding_2(val_2, dep_2, pos_2)  # [N, S_2, E // 2]

        # convolute embedded tokens of last layer
        y_0 = self.convolution_0(x_0)  # [N, S'_0, E // 4]
        # substitite all mixed token embeddings of second-last layer, with token embeddings of last layer
        x_1[val_1 == 2] = y_0[val_0[:, ::self.conv_size] != 0]  # [N, S_1, E // 4]

        # convolute substituted tokens of second-last layer
        y_1 = self.convolution_1(x_1.contiguous())  # [N, S'_1, E // 4]
        # substitite all mixed token embeddings of third-last layer, with token embeddings of second-last layer
        x_2[val_2 == 2] = y_1[val_1[:, ::self.conv_size] != 0]  # [N, S_2, E // 2]

        # convolute substituted tokens of second-last layer
        return self.convolution_2(x_2.contiguous())  # [N, S'_2, E]

    def padding_mask(self, value, depth, position):
        """ Creates a token padding mask, based on the value and depth sequence token.

        Uses only every n-th value token as input, where n is the convolution kernel size.

        Args:
            value: Value token sequence, with penultimate and last layer.
            depth: Depth token sequence, with penultimate and last layer.
            position: Position token sequence, with penultimate and last layer.

        Return:
            Padding mask, where padding tokens '0' of the value sequence are masked out.
        """
        return self.mask
