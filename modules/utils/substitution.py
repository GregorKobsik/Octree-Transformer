import torch.nn as nn
from . import Convolution


class Substitution(nn.Module):
    def __init__(self, conv_size, source_dim, target_dim):
        """ Performs a substitution operation on the input sequence vectors and convolutes them afterwards.

        Args:
            conv_size: Defines the size of the convolution kernel and stride.
            source_dim: Defines the embedding dimension of the input sequence.
            target_dim: Defines the embedding dimension of the output sequence.
        """
        super(self, Substitution).__init__()
        self.convolution = Convolution(conv_size, source_dim, target_dim)

    def forward(self, parent_vector, child_vector, mask):
        """ Replaces elements of parent vector with elements of child vector according to given mask.

        Args:
            parent_vector: Sequence vector with elements of the shape [N, P, E].
            child_vector: Sequence vector with elements of the shape [N, C, E].
            mask: Boolean vector, which defines which elements will be replaced/substituted [N, P].

        Return:
            Substituted vector with the shape [N, T, E'].
        """
        assert parent_vector.shape[0] == child_vector.shape[0], "Vectors need to have the same batch size."
        assert parent_vector.shape[2] == child_vector.shape[2], "Vectors need to have the same embedding dimension."
        assert parent_vector.shape[0:1] == mask.shape, "Mask needs to match vector shape."

        parent_vector[mask] = child_vector
        sub_vec = parent_vector.contiguous()

        return self.convolution(sub_vec)
