import math
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence
from masks import padding_mask


class DoubleConvolutionalEmbedding(nn.Module):
    def __init__(self, embed_dim, spatial_dim):
        """ Performs an embedding of token sequences into an embedding space of higher dimension.

        The embedding packs multiple tokens of the last sequence into one token embedded in higher dimension using
        convolutional operations. The encoder sequence is packed once, the decoder sequence is packed twice.
        Therefore one token in the input is responsible for multiple tokens in the output and some tokens of the input
        might be discarded in the process.

        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            embded_dim: Dimension of returned embedding space.
            spatial_dim: Spatial dimension (2D, 3D, ...) of sequence encoding.
        """
        super(DoubleConvolutionalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.spatial_dim = spatial_dim
        s = 2**spatial_dim
        conv_depth = math.sqrt(embed_dim)
        assert conv_depth == round(conv_depth), "The square root of `embed_dim` should be an integer."
        conv_depth = int(conv_depth)

        # embeddings
        self.enc_value_embedding = nn.Conv1d(1, embed_dim, kernel_size=s, stride=s)
        self.enc_depth_embedding = nn.Conv1d(1, embed_dim, kernel_size=s, stride=s)
        self.enc_position_embedding = nn.ModuleList(
            [nn.Conv1d(1, embed_dim, kernel_size=s, stride=s) for _ in range(spatial_dim)]
        )

        self.dec_value_embedding_1 = nn.Conv1d(1, conv_depth, kernel_size=1, stride=1)
        self.dec_depth_embedding_1 = nn.Conv1d(1, conv_depth, kernel_size=1, stride=1)
        self.dec_position_embedding_1 = nn.ModuleList(
            [nn.Conv1d(1, conv_depth, kernel_size=1, stride=1) for _ in range(spatial_dim)]
        )

        self.dec_value_embedding_2 = nn.Conv1d(1, conv_depth, kernel_size=s, stride=s)
        self.dec_depth_embedding_2 = nn.Conv1d(1, conv_depth, kernel_size=s, stride=s)
        self.dec_position_embedding_2 = nn.ModuleList(
            [nn.Conv1d(1, conv_depth, kernel_size=s, stride=s) for _ in range(spatial_dim)]
        )

        self.dec_latent_embedding = nn.Conv1d(conv_depth, embed_dim, kernel_size=s, stride=s)

    def source(self, value, depth, pos):
        """ Transform sequences into embedding space for the encoder.

        Uses a convolutional operation once to pack multiple tokens into one output token in higher dimension. All
        tokens are used for the embedding and none discarded.

        Args:
            value: Value token sequence for the encoder input.
            depth: Depth token sequence for the encoder input.
            position: Position token sequence for the encoder input.

        Return:
            Token sequence in the embedding space.
        """
        # cast sequences to 'float', as convolutions do not support 'long'
        value = value.float()
        depth = depth.float()
        pos = pos.float()

        # compute embeddings for all layers - [N, 1, S] -> [N, C, S']
        x = self.enc_value_embedding(torch.unsqueeze(value, dim=1))
        x = x + self.enc_depth_embedding(torch.unsqueeze(depth, dim=1))
        for axis, pos_embedding in enumerate(self.enc_position_embedding):
            x = x + pos_embedding(torch.unsqueeze(pos[:, :, axis], dim=1))

        # transpose embedding/channel dimension
        return x.transpose(1, 2)  # [N, S', C]

    def target(self, value, depth, pos):
        """ Transform sequences into embedding space for the decoder.

        Uses a convolutional operation to pack multiple tokens of the last layer into one token in higher dimension.
        All tokens of the last layer are used for the embedding and none discarded. Uses a convolutional operation to
        pack one token of the penultimate layer into one token in higher dimension and substitutes mixed ('2') token
        embeddings with the corresponding embeddings of the last layer. Therefore some tokens of the penultimate layer
        might be discarded.

        Args:
            value: Value token sequence for the decoder input, with penultimate and last layer.
            depth: Depth token sequence for the decoder input, with penultimate and last layer.
            position: Position token sequence for the decoder input, with penultimate and last layer.

        Return:
            Token sequence in the embedding space.

        TODO: Make layer splitt prettier.
        TODO: Make substitution prettier.
        TODO: Check gradients!
        """
        # cast sequences to 'float', as convolutions do not support 'long'
        value = value.float()  # [N, T]
        depth = depth.float()  # [N, T]
        pos = pos.float()  # [N, T, A]

        # create intermediate list to hold values
        batch_size = value.shape[0]

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

        # compute embeddings for penultimate layer - [N, 1, T1] -> [N, C', T1']
        x = self.dec_value_embedding_1(torch.unsqueeze(val_1, dim=1))
        x = x + self.dec_depth_embedding_1(torch.unsqueeze(dep_1, dim=1))
        for axis, pos_embedding in enumerate(self.dec_position_embedding_1):
            x = x + pos_embedding(torch.unsqueeze(pos_1[:, :, axis], dim=1))

        # compute embeddings for last layer - [N, 1, T2] -> [N, C', T2']
        y = self.dec_value_embedding_2(torch.unsqueeze(val_2, dim=1))
        y = y + self.dec_depth_embedding_2(torch.unsqueeze(dep_2, dim=1))
        for axis, pos_embedding in enumerate(self.dec_position_embedding_2):
            y = y + pos_embedding(torch.unsqueeze(pos_2[:, :, axis], dim=1))

        # flip channel and token dimensions to substitute tokens
        x = x.transpose(1, 2)  # [N, T1', C']
        y = y.transpose(1, 2)  # [N, T2', C']

        # substitite all mixed token embeddings of penultimate layer, with token embeddings of last layer
        x[val_1 == 2] = y[val_2[:, ::2**self.spatial_dim] != 0]  # [N, T1', C']

        # restore ordering of dimensions
        x = x.transpose(1, 2).contiguous()  # [N, C', T1']

        # perform embedding of the substituted latent vector into transformer latent space
        z = self.dec_latent_embedding(x)  # [N, C, T1'']

        # transpose embedding/channel dimension
        return z.transpose(1, 2)  # [N, T1'', C]

    def src_padding_mask(self, value, depth):
        """ Creates a token padding mask for the encoder, based on the value and depth sequence token.

        Args:
            value: Value token sequence for the encoder input.
            depth: Depth token sequence for the encoder input.

        Return:
            Padding mask, where padding tokens '0' of the value sequence are masked out.
        """
        # compute padding mask - check every n-th value token, where n is the size of the convolution kernel
        return padding_mask(value[:, ::2**self.spatial_dim], device=value.device)

    def tgt_padding_mask(self, value, depth):
        """ Creates a token padding mask for the decoder, based on the value and depth sequence token.

        Args:
            value: Value token sequence for the decoder input.
            depth: Depth token sequence for the decoder input.

        Return:
            Padding mask, where padding tokens '0' of the value sequence are masked out.
        """
        # initialize variables
        batch_size = value.shape[0]
        val = batch_size * [0]

        # discard last layer of input sequence
        idx = torch.argmax(depth, dim=1)
        for i in range(batch_size):
            val[i] = value[i, :idx[i]]

        # repad sequences to equal length
        val_pad = pad_sequence(val, batch_first=True, padding_value=0)

        # compute padding mask - check every n-th value token, where n is the size of the convolution kernel
        return padding_mask(val_pad[:, ::2**self.spatial_dim], device=val_pad.device)
