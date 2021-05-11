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
        for i in range(self.spatial_dim):
            elements = np.concatenate(np.split(elements, indices_or_sections=2, axis=0), axis=i)
        return np.squeeze(elements, axis=0)
