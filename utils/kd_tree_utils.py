import numpy as np
import itertools
import torch


def _directions(spatial_dim):
    return np.array(list(itertools.product([-1, 1], repeat=spatial_dim)))


class TrinaryRepresentation():
    def __init__(self, spatial_dim=3):
        """ Provides a transformation wrapper between the iterative and successive sequence format.

        Args:
            spatial_dim: Define the spatial dimensionality of the input sequences.
        """
        super().__init__()
        self.spatial_dim = spatial_dim
        self.num_tokens = 2**spatial_dim
        self.max_int_value_as_tri = 3**self.num_tokens
        self.dirs = _directions(spatial_dim)

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

    def encode_trinary(self, value, depth, position):
        """ Transforms given successive sequence into an iterative sequence representation.

        Args:
            value: Numpy array holding the value token sequence with shape (S), with token values in [0, 3]
            depth: Numpy array holding the depth token sequence with shape (S).
            position: Numpy array holding the position token sequence with shape (S, spatial_dim).

        Return:
            A tuple of (value, depth, position), where the last layer is encoded in trinary representation.
        """
        max_depth = np.max(depth)
        value_new = []
        pos_new = []

        # extract the two last layers for processing
        last_layer = value[depth == max_depth]
        penultiumate_layer = value[depth == max_depth - 1]
        penultimate_pos = position[depth == max_depth - 1]

        # iterate over the penultimate layer to encode the last layer
        for i, token in enumerate(penultiumate_layer):
            if token == 2:  # encode 8 leading tokens of last layer as one integer in trinary representation
                value_new += [self.tri_to_dec(last_layer[:self.num_tokens])]
                last_layer = last_layer[self.num_tokens:]
                pos_new += [penultimate_pos[i]]

        # discard the last layer, as we encoded it in 'target'
        value = value[depth != max_depth]
        position = position[depth != max_depth]
        depth = depth[depth != max_depth]

        # reconstruct last layer according to 'target'
        value = np.concatenate([value, value_new])
        position = np.concatenate([position, pos_new])
        depth = np.concatenate([depth, len(value_new) * [max_depth]])

        return value, depth, position

    def encode_trinary_pytorch(self, value, depth, position):
        """ Transforms given successive sequence into an iterative sequence representation. Provides a wrapper for
            pytorch tensors.

        Args:
            value: Pytorch tensor holding the value token sequence with shape (S), with token values in [0, 3].
            depth: Pytorch tensor holding the depth token sequence with shape (S).
            position: Pytorch tensor holding the position token sequence with shape (S, spatial_dim).

        Return:
            A tuple of (value, depth, position), where the last layer is encoded in trinary representation.
        """
        device = value.device

        value = value.cpu().numpy()
        depth = depth.cpu().numpy()
        position = position.cpu().numpy()

        value, depth, position = self.successive_to_iterative(value, depth, position)

        value = torch.tensor(value, dtype=torch.long, device=device)
        depth = torch.tensor(depth, dtype=torch.long, device=device)
        position = torch.tensor(position, dtype=torch.long, device=device)

        return value, depth, position

    def decode_trianry(self, value, depth, position):
        """ Transforms given iterative sequence into an successive sequence representation.

        Args:
            value: Numpy array holding the value token sequence with shape (S), with token values in [0, 3].
            depth: Numpy array holding the depth token sequence with shape (S).
            position: Numpy array holding the position token sequence with shape (S, spatial_dim).

        Return:
            A tuple of (value, depth, position), where the last layer is decoded from trinary representation.
        """
        max_depth = np.max(depth)

        value_new = np.array([])
        depth_new = np.array([])
        pos_new = np.array([])

        # compute how much does the position per token change in the next layer
        pos_step = position[0][0] // 2**(max_depth - 1)  # assume same resolution for each dimension

        # retrive values and positions of last layer from the sequence
        last_layer_value = value[depth == max_depth]
        last_layer_pos = position[depth == max_depth]

        # parse the last layer and target sequence to decode next layer
        for i in range(len(last_layer_value)):
            value_new = np.concatenate([value_new, self.dec_to_tri(last_layer_value[i])])
            depth_new = np.concatenate([depth_new, self.num_tokens * [max_depth]])
            n_pos = pos_step * self.dirs + last_layer_pos[i]
            pos_new = np.concatenate([pos_new, n_pos]) if pos_new.size != 0 else n_pos

        # discard the last layer, as we encoded it
        value = value[depth != max_depth]
        position = position[depth != max_depth]
        depth = depth[depth != max_depth]

        # concatenate sequences and return
        value = np.concatenate([value, value_new])
        depth = np.concatenate([depth, depth_new])
        position = np.concatenate([position, pos_new])

        return value, depth, position

    def decode_trinary_pytorch(self, value, depth, position):
        """ Transforms given iterative sequence into an successive sequence representation. Provides a wrapper for
            pytorch tensors.

        Args:
            value: Pytorch tensor holding the value token sequence with shape (S), with token values in [0, 3].
            depth: Pytorch tensor holding the depth token sequence with shape (S).
            position: Pytorch tensor holding the position token sequence with shape (S, spatial_dim).

        Return:
            A tuple of (value, depth, position), where the last layer is decoded from trinary representation.
        """
        device = value.device

        value = value.cpu().numpy()
        depth = depth.cpu().numpy()
        position = position.cpu().numpy()

        value, depth, position = self.iterative_to_successive(value, depth, position)

        value = torch.tensor(value, dtype=torch.long, device=device)
        depth = torch.tensor(depth, dtype=torch.long, device=device)
        position = torch.tensor(position, dtype=torch.long, device=device)

        return value, depth, position
