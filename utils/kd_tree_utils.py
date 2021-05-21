import numpy as np
import itertools
import torch


def _directions(spatial_dim):
    return np.array(list(itertools.product([-1, 1], repeat=spatial_dim)))


class RepresentationTransformator():
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

    def successive_to_iterative(self, value, depth, position):
        """ Transforms given successive sequence into an iterative sequence representation.

        Args:
            value: Numpy array holding the value token sequence with shape (S), with token values in [0, 3]
            depth: Numpy array holding the depth token sequence with shape (S).
            position: Numpy array holding the position token sequence with shape (S, spatial_dim).

        Return:
            A tuple of (value, depth, position, target), where the last depth layer is encoded in target in a trinary
            format representation.
        """
        max_depth = np.max(depth)
        target = []

        # extract the two last layers for processing
        last_layer = value[depth == max_depth]
        penultiumate_layer = value[depth == max_depth - 1]

        # iterate over the penultimate layer to encode the last layer
        for token in penultiumate_layer:
            if token == 0:  # padding token: 0
                target = target + [0]
            elif token == 1:  # token for all free: 1
                target = target + [1]
            elif token == 3:  # token for all full: 3^2^spatial_dim
                target = target + [self.max_int_value_as_tri]
            else:  # encode 8 leading tokens of last layer as one integer in trinary representation
                target += [self.tri_to_dec(last_layer[:self.num_tokens])]
                last_layer = last_layer[self.num_tokens:]

        # discard the last layer, as we encoded it in 'target'
        value = value[depth != max_depth]
        position = position[depth != max_depth]
        depth = depth[depth != max_depth]

        return value, depth, position, np.array(target)

    def successive_to_iterative_pytorch(self, value, depth, position):
        """ Transforms given successive sequence into an iterative sequence representation. Provides a wrapper for
            pytorch tensors.

        Args:
            value: Pytorch tensor holding the value token sequence with shape (S), with token values in [0, 3].
            depth: Pytorch tensor holding the depth token sequence with shape (S).
            position: Pytorch tensor holding the position token sequence with shape (S, spatial_dim).

        Return:
            A tuple of (value, depth, position, target), where the last depth layer is encoded in target in a trinary
            format representation.
        """
        device = value.device

        value = value.cpu().numpy()
        depth = depth.cpu().numpy()
        position = position.cpu().numpy()

        value, depth, position, target = self.successive_to_iterative(value, depth, position)

        value = torch.tensor(value, dtype=torch.long, device=device)
        depth = torch.tensor(depth, dtype=torch.long, device=device)
        position = torch.tensor(position, dtype=torch.long, device=device)
        target = torch.tensor(target, dtype=torch.long, device=device)

        return value, depth, position, target

    def iterative_to_successive(self, value, depth, position, target):
        """ Transforms given iterative sequence into an successive sequence representation.

        Args:
            value: Numpy array holding the value token sequence with shape (S), with token values in [0, 3].
            depth: Numpy array holding the depth token sequence with shape (S).
            position: Numpy array holding the position token sequence with shape (S, spatial_dim).
            target: Numpy array holding the value token sequence of the next layer with the shape (S), with token
                values in [0, 3**2**spatial_dim].

        Return:
            A tuple of (value, depth, position), where the target sequence is encoded in the last depth layer of the
            new sequence.
        """
        max_depth = np.max(depth)

        next_layer_value = np.array([])
        next_layer_depth = np.array([])
        next_layer_pos = np.array([])

        # compute how much does the position per token change in the next layer
        pos_step = position[0][0] // 2**(max_depth)  # assume same resolution for each dimension

        # retrive values and positions of last layer from the sequence
        last_layer_value = value[depth == max_depth]
        last_layer_pos = position[depth == max_depth]

        # parse the last layer and target sequence to decode next layer
        for i, token in enumerate(last_layer_value):
            if token == 2:
                next_layer_value = np.concatenate([next_layer_value, self.dec_to_tri(target[i])])
                n_pos = pos_step * self.dirs + last_layer_pos[i]
                next_layer_pos = np.concatenate([next_layer_pos, n_pos]) if next_layer_pos.size != 0 else n_pos
                next_layer_depth = np.concatenate([next_layer_depth, self.num_tokens * [max_depth + 1]])

        # concatenate sequences and return
        value = np.concatenate([value, next_layer_value])
        depth = np.concatenate([depth, next_layer_depth])
        position = np.concatenate([position, next_layer_pos])

        return value, depth, position

    def iterative_to_successive_pytorch(self, value, depth, position, target):
        """ Transforms given iterative sequence into an successive sequence representation. Provides a wrapper for
            pytorch tensors.

        Args:
            value: Pytorch tensor holding the value token sequence with shape (S), with token values in [0, 3].
            depth: Pytorch tensor holding the depth token sequence with shape (S).
            position: Pytorch tensor holding the position token sequence with shape (S, spatial_dim).
            target: Pytorch tensor holding the value token sequence of the next layer with the shape (S), with token
                values in [0, 3**2**spatial_dim].

        Return:
            A tuple of (value, depth, position), where the target sequence is encoded in the last depth layer of the
            new sequence.
        """
        device = value.device

        value = value.cpu().numpy()
        depth = depth.cpu().numpy()
        position = position.cpu().numpy()
        target = target.cpu().numpy()

        value, depth, position = self.iterative_to_successive(value, depth, position, target)

        value = torch.tensor(value, dtype=torch.long, device=device)
        depth = torch.tensor(depth, dtype=torch.long, device=device)
        position = torch.tensor(position, dtype=torch.long, device=device)

        return value, depth, position
