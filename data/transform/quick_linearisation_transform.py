from utils import quick_linearise


class QuickLinearisationTransform():
    def __init__(self, pos_encoding, resolution, **_):
        """Transforms voxel data into value, depth and position sequences.

        Args:
            pos_encoding: Positional encoding of the data.
            max_resolution: Select the resolution of the data.
        """
        self.pos_enc = pos_encoding
        self.max_res = resolution

    def __call__(self, voxels, **_):
        """ Perform linearisation of voxels. """
        return quick_linearise(voxels, self.pos_enc, self.max_res)
