import math
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence


class SubstitutionEmbedding(nn.Module):
    def __init__(self, num_vocab, embed_dim, resolution, spatial_dim):
        """ Performs an embedding of token sequences into an embedding space of higher dimension.

        The embedding packs multiple tokens of the last sequence into one token embedded in higher dimension using
        convolutional operations. The sequence is packed twice, therefore one token in the input is responsible for
        multiple tokens in the output and some tokens of the input might be discarded in the process.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different token values (exclusive padding token '0').
            embded_dim: Dimension of returned embedding space.
            resolution: Spatial resolution of sequence encoding.
            spatial_dim: Spatial dimension (2D, 3D, ...) of sequence encoding.
        """
        super(SubstitutionEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.spatial_dim = spatial_dim
        tree_depth = int(math.log2(resolution))
        s = 2**spatial_dim
        conv_depth = embed_dim // s

        # embeddings
        self.value_embedding_1 = nn.Embedding(num_vocab + 1, conv_depth, padding_idx=0)
        self.depth_embedding_1 = nn.Embedding(tree_depth + 1, conv_depth, padding_idx=0)
        self.spatial_embeddings_1 = nn.ModuleList(
            [nn.Embedding(2 * resolution, conv_depth, padding_idx=0) for _ in range(spatial_dim)]
        )

        self.value_embedding_2 = nn.Embedding(num_vocab + 1, conv_depth, padding_idx=0)
        self.depth_embedding_2 = nn.Embedding(tree_depth + 1, conv_depth, padding_idx=0)
        self.spatial_embeddings_2 = nn.ModuleList(
            [nn.Embedding(2 * resolution, conv_depth, padding_idx=0) for _ in range(spatial_dim)]
        )

        # convolutions
        self.conv_1 = nn.Conv1d(conv_depth, embed_dim, kernel_size=s, stride=s)
        self.conv_2 = nn.Conv1d(conv_depth, conv_depth, kernel_size=s, stride=s)

    def source(self, value, depth, pos):
        """ Transform sequences into embedding space for the encoder.

        Uses a convolutional operation to pack multiple tokens of the last layer into one token in higher dimension.
        All tokens of the last layer are used for the embedding and none discarded. Uses a convolutional operation to
        pack one token of the penultimate layer into one token in higher dimension and substitutes mixed ('2') token
        embeddings with the corresponding embeddings of the last layer. Therefore some tokens of the penultimate layer
        might be discarded.

        Args:
            value: Value token sequence for the encoder input, with penultimate and last layer.
            depth: Depth token sequence for the encoder input, with penultimate and last layer.
            pos: Position token sequence for the encoder input, with penultimate and last layer.

        Return:
            Token sequence in the embedding space.

        TODO: Make layer splitt prettier.
        TODO: Make substitution prettier.
        TODO: Check gradients!
        """
        batch_size = value.shape[0]

        # create intermediate list to hold values
        val_1 = batch_size * [0]
        dep_1 = batch_size * [0]
        pos_1 = batch_size * [0]

        val_2 = batch_size * [0]
        dep_2 = batch_size * [0]
        pos_2 = batch_size * [0]

        # splitt input in penultimate (1) and last (2) layer
        idx = torch.argmax(depth, dim=1)
        for i in range(batch_size):
            # split sequence in two layers
            val_1[i], val_2[i] = value[i, :idx[i]], value[i, idx[i]:]
            dep_1[i], dep_2[i] = depth[i, :idx[i]], depth[i, idx[i]:]
            pos_1[i], pos_2[i] = pos[i, :idx[i]], pos[i, idx[i]:]
            # unpad last layer
            pos_2[i] = pos_2[i][val_2[i] != 0]
            dep_2[i] = dep_2[i][val_2[i] != 0]
            val_2[i] = val_2[i][val_2[i] != 0]

        # repad each layer
        val_1 = pad_sequence(val_1, batch_first=True, padding_value=0)
        dep_1 = pad_sequence(dep_1, batch_first=True, padding_value=0)
        pos_1 = pad_sequence(pos_1, batch_first=True, padding_value=0)

        val_2 = pad_sequence(val_2, batch_first=True, padding_value=0)
        dep_2 = pad_sequence(dep_2, batch_first=True, padding_value=0)
        pos_2 = pad_sequence(pos_2, batch_first=True, padding_value=0)

        # compute embeddings for penultimate layer - [N, T1] -> [N, T1, C]
        x = self.value_embedding_1(val_1)
        x = x + self.depth_embedding_1(dep_1)
        for axis, spatial_embedding in enumerate(self.spatial_embeddings_1):
            x = x + spatial_embedding(pos_1[:, :, axis])

        # compute embeddings for last layer - [N, T2] -> [N, T2, C]
        y = self.value_embedding_2(val_2)
        y = y + self.depth_embedding_2(dep_2)
        for axis, spatial_embedding in enumerate(self.spatial_embeddings_2):
            y = y + spatial_embedding(pos_2[:, :, axis])

        # convolute embedded tokens of last layer
        y = self.conv_2(y.transpose(1, 2)).transpose(1, 2)  # [N, T2', C]

        # substitite all mixed token embeddings of penultimate layer, with token embeddings of last layer
        x[val_1 == 2] = y[val_2[:, ::2**self.spatial_dim] != 0]  # [N, T1, C]
        x = x.contiguous()

        # convolute embedded and substituted tokens of penultimate layer
        return self.conv_1(x.transpose(1, 2)).transpose(1, 2)  # [N, T1', E]
