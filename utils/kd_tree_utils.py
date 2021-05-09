import numpy as np


class ReprestantationTransformator():

    def __init__(self, spatial_dim=3):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.num_tokens = 2**spatial_dim
        self.max_int_value_as_tri = 3**self.num_tokens
        self.dirs = self._dirs[spatial_dim]

    def dec_to_tri(self, seq):
        """ Transformes input sequence given as a decimal number to a trinary representation as an array.

        Takes care of 0 as an additional padding value, which is reserved.
        """
        repr = np.base_repr(seq - 1, base=3, padding=self.num_tokens)[-self.num_tokens:]
        return [int(c) + 1 for c in repr]

    def tri_to_dec(self, seq):
        """ Transformes input sequence given as an integer array in trianary base to a decimal number.

        Takes care of 0 as an additional padding value, which is reserved.
        """
        repr = [c - 1 for c in seq]
        return int("".join(map(str, repr)), base=3) + 1
