import math

import torch
import torch.nn as nn


class BlockConvolution(nn.Module):
    def __init__(self, source_dim, target_dim, block_size):
        """ Performs masked blockwise convolution on an input sequence.
            The mask is always an upper right triangle matrix with zeros on the diagonal.

        Args:
            source_dim: Defines the embedding dimension of the input sequence.
            target_dim: Defines the embedding dimension of the output sequence.
            block_size: Defines the size of the block over which we convolute.
        """
        super(BlockConvolution, self).__init__()

        self.block_size = block_size
        self.convolutions = nn.ModuleList([
            nn.Conv1d(source_dim, target_dim, (i + 1,), block_size, bias=False) for i in range(block_size - 1)
        ])
        sigma = math.sqrt(1. / (block_size * source_dim))
        self.bias = nn.Parameter(torch.empty(block_size))
        nn.init.uniform_(self.bias, -sigma, sigma)

    def forward(self, seq_vector):
        """ Convolute tokens to reduce sequence length

        Args:
            seq_vector: Sequence vector with elements of the shape [N, S, E].

        Return:
            Sequence vector with the same length and target embedding dimension [N, S, E']
        """

        out = torch.zeros_like(seq_vector)
        out[:, ::self.block_size] += self.bias[0]
        for i, conv in enumerate(self.convolutions):
            out[:, 1 + i::self.block_size] = conv(seq_vector.transpose(1, 2)).transpose(1, 2)
            out[:, 1 + i::self.block_size] += self.bias[1 + i]

        return out


class BlockConvolutionLean(nn.Module):
    def __init__(self, source_dim, target_dim, block_size):
        """ Performs masked blockwise convolution on an input sequence.
            The mask is always an upper right triangle matrix with zeros on the diagonal.

        Args:
            source_dim: Defines the embedding dimension of the input sequence.
            target_dim: Defines the embedding dimension of the output sequence.
            block_size: Defines the size of the block over which we convolute.
        """
        super(BlockConvolutionLean, self).__init__()

        self.block_size = block_size
        self.convolution = nn.Conv1d(source_dim, target_dim, (1,), bias=False)
        sigma = math.sqrt(1. / (block_size * source_dim))
        self.bias = nn.Parameter(torch.empty(block_size))
        nn.init.uniform_(self.bias, -sigma, sigma)

    def forward(self, seq_vector):
        """ Convolute tokens to reduce sequence length

        Args:
            seq_vector: Sequence vector with elements of the shape [N, S, E].

        Return:
            Sequence vector with the same length and target embedding dimension [N, S, E']
        """

        features = self.convolution(seq_vector.transpose(1, 2)).transpose(1, 2)
        out = torch.zeros_like(seq_vector)
        out[:, ::self.block_size] += self.bias[0]
        for i in range(self.block_size):
            for j in range(i):
                out[:, i::self.block_size] += features[:, j::self.block_size]
            out[:, i::self.block_size] += self.bias[i]

        return out
