import numpy as np

_cmap = {
    0: 0,  # - padding value
    1: 0,  # - all pixels are empty
    2: 1,  # - pixels are empty and occupied
    3: 2,  # - all pixels are occupied
}

_dirs = np.array(
    [
        [-1, -1, -1],
        [-1, -1, +1],
        [-1, +1, -1],
        [-1, +1, +1],
        [+1, -1, -1],
        [+1, -1, +1],
        [+1, +1, -1],
        [+1, +1, +1],
    ]
)


class Octree():
    def _split(self, voxels):
        assert len(voxels.shape) == 3
        out = []
        for d in np.dsplit(voxels, 2):
            for h in np.hsplit(d, 2):
                for v in np.vsplit(h, 2):
                    out += [v]
        return np.array(out)

    def _concat(self, voxels):
        assert len(voxels) == 8
        assert len(voxels.shape) == 4
        voxels = np.concatenate([voxels[:4], voxels[4:]], axis=3)
        voxels = np.concatenate([voxels[:2], voxels[2:]], axis=2)
        voxels = np.concatenate([voxels[:1], voxels[1:]], axis=1)
        return np.squeeze(voxels, axis=0)

    def insert_voxels(self, voxels, max_depth=float('Inf'), depth=1, pos=None):
        self.depth = depth
        self.resolution = np.array(voxels.shape[0])
        self.final = True
        self.pos = np.array(voxels.shape) if pos is None else pos
        # '1' - all voxels are empty
        # '2' - voxels are empty and occupied
        # '3' - all voxels are occupied
        self.value = 1 if np.max(voxels) == 0 else 3 if np.min(voxels) > 0 else 2

        # image has a resolution of 1 and cannot be splitt anymore
        if self.resolution <= 1:
            return self

        # splitt only when voxels are mixed and we are not at maximum depth
        if self.value == 2 and depth <= max_depth:

            self.final = False
            split_voxels = self._split(voxels)

            # compute new positions for future nodes - center of all voxels
            new_pos = [self.pos + v.shape * d for v, d in zip(split_voxels, _dirs)]

            # create child nodes
            self.child_nodes = []
            for v, p in zip(split_voxels, new_pos):
                self.child_nodes += [Octree().insert_voxels(v, max_depth, depth + 1, p)]

        return self

    def get_voxels(self, depth=float('Inf'), mode='occupancy'):
        res = (self.resolution, self.resolution, self.resolution)
        if self.final or self.depth == depth:
            # return empty voxel if all voxels are empty
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

        return self._concat(np.array([node.get_voxels(depth, mode) for node in self.child_nodes]))

    def insert_sequence(self, seq, resolution, max_depth=float('Inf'), autorepair_errors=False, silent=False):
        # fail-fast: malformed input sequence
        all_tokens_valid = all(str(c) in '123' for c in seq)
        if not all_tokens_valid:
            if not silent:
                print(
                    "Error: Input sequence consists of invalid tokens. " +
                    "Valid tokens consist of 1 (white), 2 (mixed) and 3 (black). " + f"Sequence: {seq}"
                )
            raise ValueError

        # initialize parser
        depth = 1
        open_set = [self]
        pos_set = [np.array([resolution, resolution, resolution])]
        node_counter = 1
        final_layer = False

        while len(seq) > 0 and depth <= max_depth and len(open_set) > 0:
            # consume first token of sequence
            head = int(seq[0])
            seq = seq[1:] if len(seq) > 0 else seq
            node_counter -= 1

            # get next node that should be populated
            node = open_set.pop(0)

            # assign values to node
            node.value = head
            node.depth = depth
            node.resolution = np.array(resolution)
            node.pos = pos_set.pop(0)

            # final node:
            # - head is '1' or '3', thus all voxels have the same value
            # - the resolution is 1, thus the voxels cannot be split anymore
            # - we are in the last depth layer, thus all nodes are final
            node.final = head in (1, 3) or np.array_equal(resolution, [1]) or final_layer
            if not node.final:
                node.child_nodes = [Octree() for _ in range(8)]
                open_set.extend(node.child_nodes)

                # compute new positions for future nodes - center of all pixels
                pos_set.extend([node.pos + node.resolution // 2 * d for d in _dirs])

            # update depth
            if node_counter <= 0:
                depth += 1
                resolution = np.array(resolution // 2)
                # return if the resolution becomes less than 1 - no visible voxels
                if resolution < 1:
                    return self

                node_counter = len(open_set)
                # fail-fast: malformed input sequence
                if len(seq) < node_counter:
                    if not silent:
                        print(
                            "Error: Remaining input sequence is not long enough.", "Current depth:", depth,
                            "Remaining sequence: ", seq, "Current length:", len(seq), "Expected lenght:", node_counter
                        )
                    if not autorepair_errors:
                        raise ValueError
                    else:
                        # perform simple sequence repair by appending missing tokens
                        seq = np.append(seq, [0 for _ in range(node_counter - len(seq))])
                        if not silent:
                            print(f"Resolved error - Modified input sequence: {seq}, Current length: {len(seq)}")

                if len(seq) == node_counter:
                    final_layer = True

        return self

    def get_sequence(self, depth=float('Inf'), return_depth=False, return_pos=False):
        """ Returns a linearised sequence representation of the quadtree. """
        seq_value = []
        seq_depth = []
        seq_pos_x = []
        seq_pos_y = []
        seq_pos_z = []

        # start with root node
        open_set = [self]

        while len(open_set) > 0:
            node = open_set.pop(0)

            # reached sufficient depth - return sequence so far
            if node.depth > depth:
                break

            seq_value += [node.value]
            seq_depth += [node.depth]
            seq_pos_x += [node.pos[0]]
            seq_pos_y += [node.pos[1]]
            seq_pos_z += [node.pos[2]]

            if not node.final:
                open_set += node.child_nodes

        seq_value = np.asarray(seq_value)
        seq_depth = np.asarray(seq_depth)
        seq_pos_x = np.asarray(seq_pos_x)
        seq_pos_y = np.asarray(seq_pos_y)
        seq_pos_z = np.asarray(seq_pos_z)

        # output format depends in flags 'return_depth' and 'return_pos'
        if return_depth and return_pos:
            return seq_value, seq_depth, seq_pos_x, seq_pos_y, seq_pos_z
        elif return_depth and not return_pos:
            return seq_value, seq_depth
        elif not return_depth and return_pos:
            return seq_value, seq_pos_x, seq_pos_y, seq_pos_z
        else:
            return seq_value

    def __repr__(self):
        return f"Octree() = {self.get_sequence()}, len = {len(self.get_sequence())}"
