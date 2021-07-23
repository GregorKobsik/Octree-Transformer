import torch.nn as nn


class Deconvolution(nn.ConvTranspose1d):
    def __init__(self, source_dim, target_dim, conv_size):
        """ Performs a de-convolution operation on a input sequence.

        Args:
            source_dim: Defines the embedding dimension of the input sequence.
            target_dim: Defines the embedding dimension of the output sequence.
            conv_size: Defines the size of the convolution kernel and stride.
        """
        super(self, Deconvolution).__init__(source_dim, target_dim, kernel_size=conv_size, stride=conv_size)

    def forward(self, seq_vector):
        """ Deconvolute tokens to reduce sequence length

        Args:
            seq_vector: Sequence vector with elements of the shape [N, S, E].

        Return:
            Sequence vector with reduced length and target embedding dimension [N, T, E']
        """
        return super().forward(seq_vector.transpose(1, 2)).transpose(1, 2)
