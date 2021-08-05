import numpy as np
from utils import _directions, concat, split

_cmap = {
    0: 0,  # - padding value, show as empty
    1: 0,  # - all elements are empty
    2: 1,  # - elements are undefined, empty and occupied
    3: 2,  # - all elements are occupied
}


class kdTree():
    """ Implements a kd-tree data structure for volumetric/spatial objects. Works with arrays of spatial data as well
    as linearised token sequence representations.

    This class allows to transform array with spatial elements into kd-trees, where k can by any natural number. Each
    node represents mixed elements, which can be split in its branches. Each leaf represents a final element which is
    either completly empty or completly occupied. These structure can be than linearized as a sequence of tokens, which
    is equivalent to the kd-tree. In the same way, as arrays with spatial elements can be transformed into kd-trees,
    token sequences can be transformed into kd-trees. This allows to seamlessly transform arrays of spatial data into
    token sequences and vice versa.
    """
    def __init__(self, spatial_dim: int, pos_encoding: str = "centered"):
        """ Initializes the kd-tree for the right spatial dimensionality.

        Args:
            spatial_dim: Defines the spatial dimensionality of the kd-tree, e.g. '2' for images/pixels and '3' for
                volumes/voxels.
            pos_encoding: Defines the positional encoding of positions. It uses either a centered position,
                where each position relates to the center of all pixels/voxels or an intertwined encoding, where each
                layer uses an ascending, axis aligned enumeration, thus the position values are intertwined.
        """
        super().__init__()
        self.spatial_dim = spatial_dim
        self.intertwined_positions = pos_encoding == 'intertwined'
        self.pos_encoding = pos_encoding
        self.dirs = _directions(spatial_dim, pos_encoding)

    def insert_element_array(self, elements, max_depth=float('Inf'), depth=0, pos=None):
        """ Inserts an array of element values which is converted into a kd-tree.

        Args:
            elements: A numpy array of element values, with the dimensionality of the kd-tree.
            max_depth: The maximum depth of the resulting kd-tree. All nodes at `max_depth` are marked as final.
            depth: The current depth of the kd-tree. Used to recursively define the tree depth.
            pos: Defines the mean position of all elements at the current node.

        Return:
            The current node containing inserted values. The returned node should be the root node of the kd-tree.
        """
        self.depth = depth
        self.resolution = np.array(elements.shape[0])
        self.final = True
        if self.intertwined_positions:
            self.pos = np.array(self.spatial_dim * [0]) if pos is None else pos
        else:
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
            sub_elements = split(elements)

            # compute new positions for future nodes
            if self.intertwined_positions:
                # layerwise intertwined_positions
                new_pos = [2 * self.pos + d for d in self.dirs]
            else:
                # center of all sub elements
                new_pos = [self.pos + e.shape * d for e, d in zip(sub_elements, self.dirs)]

            # create child nodes
            self.child_nodes = [
                kdTree(self.spatial_dim, self.pos_encoding).insert_element_array(e, max_depth, depth + 1, p)
                for e, p in zip(sub_elements, new_pos)
            ]

        return self

    def get_element_array(self, depth=float('Inf'), mode='occupancy'):
        """ Converts the kd-tree into an array of elements.

        Args:
            depth: Defines the maximum depth of the children nodes, of which the value will be returned in the array.
            mode: Defines how the value of each node should be represented in the returned array. `occupancy` - returns
                all padding and empty values as '0' and all mixed and occupied values as '1'. `value` - return the
                exact value stored in the node. `color` - returns the values based on a colormap defined in `_cmap`,
                where the stored value is subtracted by 1 and the padding value is returned as '0'. `depth` - returns
                the current depth of the node as value in the array. `random` - returns a random number in the range of
                [0, 19] for each node.

        Return:
            A numpy array with the dimensionality of the kd-tree, which hold values defined by `mode`.

        """
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

        return concat(np.array([node.get_element_array(depth, mode) for node in self.child_nodes]))

    def insert_token_sequence(self, value, resolution, max_depth=float('Inf'), autorepair_errors=False, silent=False):
        """ Inserts a token sequence which is parsed into a kd-tree.

        Args:
            value: A token sequence representing a spatial object. The values should consist only of '1', '2' and '3'.
                The sequence can be eiter a string or an array of strings or integers.
            resolution: The resolution of the token sequence. This value should be a power of 2.
            max_depth: The maximum depth up to which the token sequence will be parsed.
            autorepair_errors: Select if the parser should try to automatically repair malformed input sequenced by
                adding padding tokens up to a required length. Each node with a value of '2' should have
                2**`spatial_dim` children nodes.
            silent: Select if errors and warnings should be printed into the output console.

        Return:
            A node which represents the given token sequence. The returned node should be the root node of the kd-tree.
        """
        # fail-fast: malformed input sequence
        all_tokens_valid = all(str(c) in '123' for c in value)
        if not all_tokens_valid:
            raise ValueError(
                "ERROR: Input sequence consists of invalid tokens. Check token values and array type." +
                f"Valid tokens consist of 1 (white), 2 (mixed) and 3 (black). Sequence: {value}."
            )

        # initialize self
        self.value = 0
        self.depth = 0
        if self.intertwined_positions:
            self.pos = np.array(self.spatial_dim * [0])
        else:
            self.pos = np.array(self.spatial_dim * [resolution])
        self.resolution = np.array(resolution)
        self.final = False

        # initialize parser
        depth = 1
        final_layer = False
        resolution = resolution // 2

        # initialize first nodes
        open_set = []
        self.child_nodes = [kdTree(self.spatial_dim, self.pos_encoding) for _ in range(2**self.spatial_dim)]
        open_set.extend(self.child_nodes)
        node_counter = len(open_set)

        # compute new positions for future nodes
        if self.intertwined_positions:
            # layerwise intertwined_positions
            pos_set = [2 * self.pos + d for d in self.dirs]
        else:
            # center of all pixels
            pos_set = [self.pos + self.resolution // 2 * d for d in self.dirs]

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
            node.pos = pos_set.pop(0)
            node.resolution = np.array(resolution)

            # final node:
            # - head is '1' or '3', thus all elements have the same value
            # - the resolution is 1, thus the elements cannot be split anymore
            # - we are in the last depth layer, thus all nodes are final
            node.final = head in (1, 3) or np.array_equal(resolution, [1]) or final_layer
            if not node.final:
                node.child_nodes = [kdTree(self.spatial_dim, self.pos_encoding) for _ in range(2**self.spatial_dim)]
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
                            "WARNING: Remaining input sequence is not long enough.", "Current depth:", depth,
                            "Remaining sequence: ", value, "Current length:", len(value), "Expected lenght:",
                            node_counter
                        )
                    if not autorepair_errors:
                        print("ERROR: Malformed input sequence not resolved.")
                        raise ValueError
                    else:
                        # perform simple sequence repair by appending missing tokens
                        value = np.append(value, [0 for _ in range(node_counter - len(value))])
                        if not silent:
                            print(
                                f"WARNING: Resolved error - Modified input sequence: {value}, " +
                                f"Current length: {len(value)}"
                            )

                if len(value) == node_counter:
                    final_layer = True

        return self

    def get_token_sequence(self, depth=float('Inf'), return_depth=False, return_pos=False):
        """ Returns a linearised sequence representation of the kd-tree.

        Args:
            depth: Defines the maximum depth of the nodes, up to which the tree is parsed.
            return_depth: Selects if the corresponding depth sequence should be returned.
            return_pos: Selects if the corresponding position sequence should be returned.

        Return
            A numpy array consisting of integer values representing the linearised kd-tree. Returns additionally the
            corresponding depth and position sequence if specified in `return_depth` or `return_pos`. The values are
            returned in the following order: (value, depth, position).
        """
        seq_value = []
        seq_depth = []
        seq_pos = []
        open_set = []

        # start with root node
        open_set.extend(self.child_nodes)

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
        output = [seq_value]
        if return_depth:
            output += [seq_depth]
        if return_pos:
            output += [seq_pos]
        return output

    def __repr__(self):
        """ Returns of human readable string representation of the kd-tree. """
        return (
            f"kdTree() = {self.get_token_sequence()[0]}, " + f"len = {len(self.get_token_sequence()[0])}, " +
            f"dim = {self.spatial_dim}"
        )
