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

