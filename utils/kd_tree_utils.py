import numpy as np
import itertools
import torch


def _directions(spatial_dim):
    return np.array(list(itertools.product([-1, 1], repeat=spatial_dim)))


class TrinaryRepresentation():
    def __init__(self, spatial_dim=3):
        """ Provides a transformation wrapper between the basic and trinary sequence format.

        Args:
            spatial_dim: Define the spatial dimensionality of the input sequences.
        """
        super().__init__()
        self.spatial_dim = spatial_dim
        self.num_tokens = 2**spatial_dim
        self.max_int_value_as_tri = 3**self.num_tokens
        self.dirs = _directions(spatial_dim)

    def dec_to_tri(self, seq):
        """ Transformes input sequence given as a single decimal number to a trinary representation as an array.

        Takes care of `0` as an additional padding value, which is reserved.
        """
        repr = np.base_repr(seq - 1, base=3, padding=self.num_tokens)[-self.num_tokens:]
        return [int(c) + 1 for c in repr]

    def tri_to_dec(self, seq):
        """ Transformes input sequence given as an integer array in trianary base to a single decimal number.

        Takes care of `0` as an additional padding value, which is reserved.
        """
        repr = [c - 1 for c in seq]
        return int("".join(map(str, repr)), base=3) + 1

    def encode_trinary(self, value, depth, position):
        """ Transforms given basic sequence into a trinary sequence representation.

        Args:
            value: Numpy array holding the value token sequence with shape [S], with token values in [0, 3]
            depth: Numpy array holding the depth token sequence with shape [S].
            position: Numpy array holding the position token sequence with shape [S, spatial_dim].

        Return:
            A tuple of (value, depth, position) in trinary representation.
        """
        value_trinary = []

        # reshape value tokens into tuples, where one tuple represents exactly one new token
        value = value.reshape(-1, self.num_tokens)

        # iterate over the value sequence to encode tuples into tokens in trinary representation
        for token in value:
            value_trinary += [self.tri_to_dec(token)]

        # recompute positions: compute the mean position of each n tokens.
        position = position.reshape(-1, self.num_tokens, self.spatial_dim)
        position = position.sum(axis=1) // self.num_tokens

        # take each n-th depth token, as we summarized them
        depth = depth[::self.num_tokens]

        return np.array(value_trinary), depth, position

    def encode_trinary_pytorch(self, value, depth, position):
        """ Transforms given basic sequence into a trinary sequence representation. Provides a wrapper for pytorch
            tensors.

        Args:
            value: Pytorch tensor holding the value token sequence with shape (S), with token values in [0, 3].
            depth: Pytorch tensor holding the depth token sequence with shape (S).
            position: Pytorch tensor holding the position token sequence with shape (S, spatial_dim).

        Return:
            A tuple of (value, depth, position) in trinary representation.
        """
        # remember on which device the sequences are stored
        device = value.device

        # move sequences to cpu and convert them to numpy arrays
        value = value.cpu().numpy()
        depth = depth.cpu().numpy()
        position = position.cpu().numpy()

        # transform value, depth and position sequences
        value, depth, position = self.encode_trinary(value, depth, position)

        # recast numpy arrays into pytorch tensors and move to original device
        value = torch.tensor(value, dtype=torch.long, device=device)
        depth = torch.tensor(depth, dtype=torch.long, device=device)
        position = torch.tensor(position, dtype=torch.long, device=device)

        return value, depth, position

    def decode_trinary_value(self, value):
        """ Transforms given trinary value sequence into a basic sequence representation.

        Args:
            value: Numpy array holding the value token sequence with shape (S), with token values in [0, 3].

        Return:
            Value sequence in basic sequence representation.
        """
        # iterate over the value sequence to decode a single token into multiple basic tokens
        value_new = []
        for val_token in value:
            value_new += self.dec_to_tri(val_token)
        value = np.array(value_new).reshape(-1)

        return value

    def decode_trinary(self, value, depth, position):
        """ Transforms given trinary sequence into a basic sequence representation.

        Args:
            value: Numpy array holding the value token sequence with shape (S), with token values in [0, 3].
            depth: Numpy array holding the depth token sequence with shape (S).
            position: Numpy array holding the position token sequence with shape (S, spatial_dim).

        Return:
            A tuple of (value, depth, position) in basic sequence representation.
        """
        # decode a value sequence in trinary representation into basic representation
        value = self.decode_trinary_value(value)

        # create n-times as much depth tokens, as we decoded each value token into multiple ones
        depth = np.repeat(depth, self.num_tokens)

        # compute the absolute position delta for each token - ssume same resolution in each spatial dimension
        pos_steps = np.repeat(position[0][0] // 2**(depth), self.spatial_dim).reshape(-1, self.spatial_dim)
        # precompute position deltas for each token
        pos_deltas = np.tile(self.dirs, (len(position), 1)) * pos_steps
        # compute new position values for each generated token
        position = np.repeat(position, self.num_tokens, axis=0) + pos_deltas

        return value, depth, position

    def decode_trinary_pytorch(self, value, depth, position):
        """ Transforms given trinary sequence into a basic sequence representation. Provides a wrapper for pytorch
            tensors.

        Args:
            value: Pytorch tensor holding the value token sequence with shape (S), with token values in [0, 3].
            depth: Pytorch tensor holding the depth token sequence with shape (S).
            position: Pytorch tensor holding the position token sequence with shape (S, spatial_dim).

        Return:
            A tuple of (value, depth, position) in basic sequence representation.
        """
        # remember on which device the sequences are stored
        device = value.device

        # move sequences to cpu and convert them to numpy arrays
        value = value.cpu().numpy()
        depth = depth.cpu().numpy()
        position = position.cpu().numpy()

        # transform value, depth and position sequences
        value, depth, position = self.decode_trinary(value, depth, position)

        # recast numpy arrays into pytorch tensors and move to original device
        value = torch.tensor(value, dtype=torch.long, device=device)
        depth = torch.tensor(depth, dtype=torch.long, device=device)
        position = torch.tensor(position, dtype=torch.long, device=device)

        return value, depth, position
