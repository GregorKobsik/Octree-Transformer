import math
import torch
import torch.nn as nn


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
        max_depth = torch.max(depth)
        len_1 = torch.sum(depth == (max_depth - 1), dim=1)
        len_2 = torch.sum(depth == max_depth, dim=1)

        # create intermediate list to hold values
        val_1 = torch.zeros((batch_size, max(len_1)), dtype=torch.long, device=value.device)
        val_2 = torch.zeros((batch_size, max(len_2)), dtype=torch.long, device=value.device)

        # splitt input in penultimate (1) and last (2) layer
        for i in range(batch_size):
            val_1[i, :len_1[i]] = value[i, :len_1[i]]
            val_2[i, :len_2[i]] = value[i, len_1[i]:len_1[i] + len_2[i]]

        # compute embeddings
        x = self.value_embedding_1(val_1)  # [N, T1] -> [N, T1, C]
        y = self.value_embedding_2(val_2)  # [N, T2] -> [N, T2, C]

        # convolute embedded tokens of last layer
        y = self.conv_2(y.transpose(1, 2)).transpose(1, 2)  # [N, T2', C]

        # substitite all mixed token embeddings of penultimate layer, with token embeddings of last layer
        x[val_1 == 2] = y[val_2[:, ::2**self.spatial_dim] != 0]  # [N, T1, C]
        x = x.contiguous()

        # convolute substituted tokens of penultimate layer
        return self.conv_1(x.transpose(1, 2)).transpose(1, 2)  # [N, T1', E]
