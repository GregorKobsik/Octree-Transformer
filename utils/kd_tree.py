import numpy as np
import itertools

_cmap = {
    0: 0,  # - padding value
    1: 0,  # - all pixels are empty
    2: 1,  # - pixels are empty and occupied
    3: 2,  # - all pixels are occupied
}


def _directions(spatial_dim):
    return np.array(list(itertools.product([-1, 1], repeat=spatial_dim)))


class kdTree():
    def __init__(self, spatial_dim: int):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.dirs = _directions(spatial_dim)

    def _split(self, elements):
        elements = np.expand_dims(elements, axis=0)
        for i in range(self.spatial_dim, 0, -1):
            elements = np.concatenate(np.split(elements, indices_or_sections=2, axis=i), axis=0)
        return elements

    def _concat(self, elements):
        for i in range(1, self.spatial_dim + 1):
            elements = np.concatenate(np.split(elements, indices_or_sections=2, axis=0), axis=i)
        return np.squeeze(elements, axis=0)

    def insert_element_array(self, elements, max_depth=float('Inf'), depth=1, pos=None):
        self.depth = depth
        self.resolution = np.array(elements.shape[0])
        self.final = True
        self.pos = np.array(elements.shape) if pos is None else pos
        # '1' - all elements are empty
        # '2' - elements are empty and occupied
        # '3' - all elements are occupied
        self.value = 1 if np.max(elements) == 0 else 3 if np.min(elements) > 0 else 2

        # input has a resolution of 1 and cannot be splitt anymore
        if self.resolution <= 1:
            return self

        # splitt only when elements are mixed and we are not at maximum depth
        if self.value == 2 and depth <= max_depth:
            self.final = False

            # split elements into subarrays
            sub_elements = self._split(elements)

            # compute new positions for future nodes - center of all sub elements
            new_pos = [self.pos + e.shape * d for e, d in zip(sub_elements, self.dirs)]

            # create child nodes
            self.child_nodes = [
                kdTree(self.spatial_dim).insert_element_array(e, max_depth, depth + 1, p)
                for e, p in zip(sub_elements, new_pos)
            ]

        return self

    def get_element_array(self, depth=float('Inf'), mode='occupancy'):
        res = self.spatial_dim * [self.resolution]
        if self.final or self.depth == depth:
            # return empty array if all elements are empty
            if self.value == 1:
                return np.tile(0, res)
            # else return value based on `mode`
            elif mode == 'occupancy':
                return np.tile(1, res)
            elif mode == 'value':
                return np.tile(self.value, res)
            elif mode == 'color':
                return np.tile(_cmap[self.value], res)
            elif mode == 'depth':
                return np.tile(self.depth, res)
            elif mode == 'random':
                return np.tile((np.random.rand(1) * 20) % 20, res)

        return self._concat(np.array([node.get_element_array(depth, mode) for node in self.child_nodes]))

    def insert_token_sequence(self, value, resolution, max_depth=float('Inf'), autorepair_errors=False, silent=False):
        # fail-fast: malformed input sequence
        all_tokens_valid = all(str(c) in '123' for c in value)
        if not all_tokens_valid:
            if not silent:
                print(
                    "Error: Input sequence consists of invalid tokens. " +
                    f"Valid tokens consist of 1 (white), 2 (mixed) and 3 (black). Sequence: {value}."
                )
            raise ValueError

        # initialize parser
        depth = 1
        open_set = [self]
        pos_set = [np.array(self.spatial_dim * [resolution])]
        node_counter = 1
        final_layer = False

        while len(value) > 0 and depth <= max_depth and len(open_set) > 0:
            # consume first token of sequence
            head = int(value[0])
            value = value[1:] if len(value) > 0 else value
            node_counter -= 1

            # get next node that should be populated
            node = open_set.pop(0)

            # assign values to node
            node.value = head
            node.depth = depth
            node.resolution = np.array(resolution)
            node.pos = pos_set.pop(0)

            # final node:
            # - head is '1' or '3', thus all elements have the same value
            # - the resolution is 1, thus the elements cannot be split anymore
            # - we are in the last depth layer, thus all nodes are final
            node.final = head in (1, 3) or np.array_equal(resolution, [1]) or final_layer
            if not node.final:
                node.child_nodes = [kdTree(self.spatial_dim) for _ in range(2**self.spatial_dim)]
                open_set.extend(node.child_nodes)

                # compute new positions for future nodes - center of all pixels
                pos_set.extend([node.pos + node.resolution // 2 * d for d in self.dirs])

            # update depth
            if node_counter <= 0:
                depth += 1
                resolution = np.array(resolution // 2)
                # return if the resolution becomes less than 1 - no visible elements
                if resolution < 1:
                    return self

                node_counter = len(open_set)
                # fail-fast: malformed input sequence
                if len(value) < node_counter:
                    if not silent:
                        print(
                            "Error: Remaining input sequence is not long enough.", "Current depth:", depth,
                            "Remaining sequence: ", value, "Current length:", len(value), "Expected lenght:",
                            node_counter
                        )
                    if not autorepair_errors:
                        raise ValueError
                    else:
                        # perform simple sequence repair by appending missing tokens
                        value = np.append(value, [0 for _ in range(node_counter - len(value))])
                        if not silent:
                            print(f"Resolved error - Modified input sequence: {value}, Current length: {len(value)}")

                if len(value) == node_counter:
                    final_layer = True

        return self

    def get_token_sequence(self, depth=float('Inf'), return_depth=False, return_pos=False):
        """ Returns a linearised sequence representation of the quadtree. """
        seq_value = []
        seq_depth = []
        seq_pos = []

        # start with root node
        open_set = [self]

        while len(open_set) > 0:
            node = open_set.pop(0)

            # reached sufficient depth - return sequence so far
            if node.depth > depth:
                break

            seq_value += [node.value]
            seq_depth += [node.depth]
            seq_pos += [node.pos]

            if not node.final:
                open_set += node.child_nodes

        seq_value = np.asarray(seq_value)
        seq_depth = np.asarray(seq_depth)
        seq_pos = np.asarray(seq_pos)

        # output format depends in flags 'return_depth' and 'return_pos'
        output = seq_value
        if return_depth:
            output = output + (seq_depth)
        if return_pos:
            output = output + (seq_pos)
        return output

    def __repr__(self):
        return (
            f"kdTree() = {self.get_token_sequence()}, " + f"len = {len(self.get_token_sequence())}, " +
            f"dim = {self.spatial_dim}"
        )
