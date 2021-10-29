import torch.nn as nn


class Linear(nn.Linear):
    def __init__(self, embed_dim, num_vocab):
        """ Performs a convolution operation on a input sequence.

        Args:
            embed_dim: Dimension of returned embedding space.
            num_vocab: Number of different token values (exclusive padding token '0').
        """
        super(Linear, self).__init__(embed_dim, num_vocab + 1, bias=True)
