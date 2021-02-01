import numpy as np

from operator import add
from functools import reduce


class QuadTree():
    def _split4(self, image):
        half_split = np.array_split(image, 2)
        res = map(lambda x: np.array_split(x, 2, axis=1), half_split)
        return reduce(add, res)

    def _concatenate4(self, north_west, north_east, south_west, south_east):
        top = np.concatenate((north_west, north_east), axis=1)
        bottom = np.concatenate((south_west, south_east), axis=1)
        return np.concatenate((top, bottom), axis=0)

    def insert_image(self, img, mode='mean', max_depth=float('Inf'), depth=0):
        self.depth = depth
        self.resolution = img.shape
        self.final = True
        if mode == 'mean':
            self.value = np.mean(img, axis=(0, 1))
        elif mode == 'median':
            self.value = np.median(img, axis=(0, 1))
        elif mode == 'max':
            self.value = np.max(img, axis=(0, 1))
        elif mode == 'binary':
            self.value = 1 if np.max(img, axis=(0, 1)) > 0 else 0

        if depth < max_depth:
            if np.any(img > 0):  # check split
                if np.any(np.asarray(img.shape) <= 1, axis=-1):
                    return self  # image has width or height of 1 and thus is cannot be splitted

                split_img = self._split4(img)

                self.final = False
                self.north_west = QuadTree().insert_image(split_img[0], mode, max_depth, depth + 1)
                self.north_east = QuadTree().insert_image(split_img[1], mode, max_depth, depth + 1)
                self.south_west = QuadTree().insert_image(split_img[2], mode, max_depth, depth + 1)
                self.south_east = QuadTree().insert_image(split_img[3], mode, max_depth, depth + 1)

        return self

    def get_image(self, depth=float('Inf'), mode='value'):
        if (self.final or self.depth == depth):
            if mode == 'value':
                return np.tile(self.value, self.resolution)
            elif mode == 'depth':
                return np.tile(self.depth, self.resolution)
            elif mode == 'random':
                return np.tile(np.random.rand(1), self.resolution)

        return self._concatenate4(
            self.north_west.get_image(depth, mode),
            self.north_east.get_image(depth, mode),
            self.south_west.get_image(depth, mode),
            self.south_east.get_image(depth, mode),
        )

    def insert_sequence(self, seq, resolution, max_depth=float('Inf'), autorepair_errors=False, silent=False):
        # fail-fast: malformed input sequence
        all_binary = all(str(c) in '01' for c in seq)
        if not all_binary:
            if not silent:
                print("Error: Input sequence is not in binary format.", "Sequence:", seq)
            raise ValueError

        # initialize parser
        depth = 0
        open_set = []
        open_set.append(self)
        node_counter = 1
        final_node = False

        while len(seq) > 0 and depth < max_depth and len(open_set) > 0:
            # consume first token of sequence
            head = int(seq[0])
            seq = seq[1:] if len(seq) > 0 else seq

            # get next node that should be populated
            node = open_set.pop(0)

            # assign values to node
            node_counter -= 1
            node.value = head
            node.depth = depth
            node.resolution = resolution

            # final node: if head is `0` or we are in the last depth layer or the node is final
            node.final = final_node or not head or resolution == (1, 1)
            # branching: if head is `1` and we are NOT in the last depth layer, add new nodes
            if not final_node and head:
                node.north_west = QuadTree()
                node.north_east = QuadTree()
                node.south_west = QuadTree()
                node.south_east = QuadTree()
                open_set.extend([node.north_west, node.north_east, node.south_west, node.south_east])

            # update depth
            if node_counter <= 0:
                depth += 1
                # TODO: only assumes quadratic resolution with edge length 2 ** n
                resolution = tuple(int(i / 2) for i in resolution)
                node_counter = len(open_set)
                if len(seq) < node_counter:  # fail-fast: malformed input sequence
                    if not silent:
                        print(
                            "Error: Remaining input sequence is not long enough.", "Abort at depth:", depth,
                            "Remaining sequence: ", seq, "Current length:", len(seq), "Expected lenght:", node_counter
                        )
                    if not autorepair_errors:
                        raise ValueError
                    else:
                        # perform simple sequence repair by appending missing tokens
                        seq = np.append(seq, [1 for _ in range(node_counter - len(seq))])
                        if not silent:
                            print(f"Resolved error - Modified input sequence: {seq}, Current length: {len(seq)}")
                if len(seq) == node_counter:
                    final_node = True

        return self

    def get_sequence(self, depth=float('Inf'), as_string=False):

        seq = "" if as_string else []
        open_set = []
        open_set.append(self)  # push root node to open set

        while len(open_set) > 0:
            node = open_set.pop(0)  # get first node in open set

            if node.depth > depth:
                return seq  # reached sufficient deapth - return sequence so far

            val = int(np.any(node.value > 0))  # `1` if the value is positive, otherwise `0`
            seq += str(val) if as_string else [val]

            # append children if value was positive and not the last node
            if np.any(node.value > 0) and not node.final:
                open_set.append(node.north_west)
                open_set.append(node.north_east)
                open_set.append(node.south_west)
                open_set.append(node.south_east)

        return seq

    def __repr__(self):
        return f"QuadTree() = {self.get_sequence()}, len = {len(self.get_sequence())}"
