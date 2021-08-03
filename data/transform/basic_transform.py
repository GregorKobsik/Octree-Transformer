import math
from utils import kdTree


class BasicTransform():
    def __init__(self, spatial_dim, position_encoding, resolution, **_):
        """ Creates a transform module, which transforms the input data samples to pytorch tensors.

        Args:
            spatial_dim: Spatial dimensionality of input data.
            position_encoding: Positional encoding of the data.
            resolution: Select the resolution of the data.
        """
        self.spatial_dim = spatial_dim
        self.position_encoding = position_encoding
        self.resolution = resolution

    def __call__(self, voxels, **_):
        """ Transforms a single sample to pytorch tensors. """
        octree = kdTree(self.spatial_dim, self.position_encoding).insert_element_array(voxels)
        return octree.get_token_sequence(
            depth=math.log2(self.resolution),
            return_depth=True,
            return_pos=True,
        )
