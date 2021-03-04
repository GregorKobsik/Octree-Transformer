import numpy as np

from operator import add
from functools import reduce

_cmap = {
    0: 0.7,  # - padding value
    1: 1.0,  # - all pixels are white
    2: 0.3,  # - pixels are white and black
    3: 0.0,  # - all pixels are black
}


class Quadtree():
    def _split4(self, image):
        half_split = np.array_split(image, 2)
        res = map(lambda x: np.array_split(x, 2, axis=1), half_split)
        return reduce(add, res)

    def _concatenate4(self, north_west, north_east, south_west, south_east):
        top = np.concatenate((north_west, north_east), axis=1)
        bottom = np.concatenate((south_west, south_east), axis=1)
        return np.concatenate((top, bottom), axis=0)

    def insert_image(self, img, max_depth=float('Inf'), depth=1, pos=None):
        self.depth = depth
        self.resolution = np.array(img.shape)
        self.final = True
        self.pos = np.array(img.shape) if pos is None else pos
        # '1' - all pixels are white
        # '2' - pixels are white and black
        # '3' - all pixels are black
        self.value = 1 if np.max(img) == 0 else 3 if np.min(img) > 0 else 2

        # splitt only when pixels are mixed and we are not at maximum depth
        if self.value == 2 and depth <= max_depth:

            # image has width or height of 1 and thus cannot be split anymore
            if np.any(np.array(img.shape) <= 1):
                return self

            split_img = self._split4(img)

            # compute new positions for future nodes - center of all pixels
            delta_pos = np.array([[-1, -1], [+1, -1], [-1, 1], [+1, +1]])
            new_pos = [self.pos + img.shape * delta for img, delta in zip(split_img, delta_pos)]

            self.final = False
            self.north_west = Quadtree().insert_image(split_img[0], max_depth, depth + 1, new_pos[0])
            self.north_east = Quadtree().insert_image(split_img[1], max_depth, depth + 1, new_pos[1])
            self.south_west = Quadtree().insert_image(split_img[2], max_depth, depth + 1, new_pos[2])
            self.south_east = Quadtree().insert_image(split_img[3], max_depth, depth + 1, new_pos[3])

        return self

    def get_image(self, depth=float('Inf'), mode='value'):
        if (self.final or self.depth == depth):
            if mode == 'value':
                return np.tile(self.value, self.resolution)
            elif mode == 'color':
                return np.tile(_cmap[self.value], self.resolution)
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

    def insert_sequence(self, seq_value, resolution, max_depth=float('Inf'), autorepair_errors=False, silent=False):
        # fail-fast: malformed input sequence
        all_tokens_valid = all(str(c) in '123' for c in seq_value)
        if not all_tokens_valid:
            if not silent:
                print(
                    "Error: Input sequence consists of invalid tokens. " +
                    "Valid tokens consist of 1 (white), 2 (mixed) and 3 (black). " + f"Sequence: {seq_value}"
                )
            raise ValueError

        # initialize parser
        depth = 1
        open_set = [self]
        pos_set = [np.array(resolution)]
        node_counter = 1
        final_layer = False

        while len(seq_value) > 0 and depth <= max_depth and len(open_set) > 0:
            # consume first token of sequence
            head = int(seq_value[0])
            seq_value = seq_value[1:] if len(seq_value) > 0 else seq_value
            node_counter -= 1

            # get next node that should be populated
            node = open_set.pop(0)

            # assign values to node
            node.value = head
            node.depth = depth
            node.resolution = np.array(resolution)
            node.pos = pos_set.pop(0)

            # final node:
            # - head is '1' or '3', thus all pixels have the same color
            # - the resolution is (1, 1), thus the image cannot be split anymore
            # - we are in the last depth layer, thus all nodes are final
            node.final = head in (1, 3) or np.array_equal(resolution, [1, 1]) or final_layer
            if not node.final:
                node.north_west = Quadtree()
                node.north_east = Quadtree()
                node.south_west = Quadtree()
                node.south_east = Quadtree()
                open_set.extend([node.north_west, node.north_east, node.south_west, node.south_east])

                # compute new positions for future nodes - center of all pixels
                delta_pos = [[-1, -1], [+1, -1], [-1, 1], [+1, +1]]
                pos_set.extend([node.pos + node.resolution // 2 * delta for delta in delta_pos])

            # update depth
            if node_counter <= 0:
                depth += 1
                # TODO: only assumes quadratic resolution with edge length 2 ** n
                resolution = np.array([i // 2 for i in resolution])
                node_counter = len(open_set)

                # fail-fast: malformed input sequence
                if len(seq_value) < node_counter:
                    if not silent:
                        print(
                            f"Error: Remaining input sequence is not long enough. Current depth: {depth}, ",
                            f"Remaining sequence: {seq_value}, Current length: {len(seq_value)}, ",
                            f"Expected lenght: {node_counter}."
                        )
                    if not autorepair_errors:
                        raise ValueError
                    else:
                        # perform simple sequence repair by appending missing tokens
                        seq_value = np.append(seq_value, [0 for _ in range(node_counter - len(seq_value))])
                        if not silent:
                            print(
                                f"Resolved error - Modified input sequence: {seq_value}, ",
                                f"Current length: {len(seq_value)}"
                            )

                if len(seq_value) == node_counter:
                    final_layer = True

        return self

    def get_sequence(self, depth=float('Inf'), return_depth=False, return_pos=False):
        """ Returns a linearised sequence representation of the quadtree. """
        seq_value = []
        seq_depth = []
        seq_pos_x = []
        seq_pos_y = []

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

            if not node.final:
                open_set += [node.north_west]
                open_set += [node.north_east]
                open_set += [node.south_west]
                open_set += [node.south_east]

        seq_value = np.asarray(seq_value)
        seq_depth = np.asarray(seq_depth)
        seq_pos_x = np.asarray(seq_pos_x)
        seq_pos_y = np.asarray(seq_pos_y)

        # output format depends in flags 'return_depth' and 'return_pos'
        if return_depth and return_pos:
            return seq_value, seq_depth, seq_pos_x, seq_pos_y
        elif return_depth and not return_pos:
            return seq_value, seq_depth
        elif not return_depth and return_pos:
            return seq_value, seq_pos_x, seq_pos_y
        else:
            return seq_value

    def __repr__(self):
        return f"Quadtree() = {self.get_sequence()}, len = {len(self.get_sequence())}"
