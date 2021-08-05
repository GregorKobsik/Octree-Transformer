from utils import piecewise_linear_warping


class PiecewiseLinearWarpingTransform():
    def __init__(self, **_):
        """Scales input data for each axis in the range of [0.75 .. 1.25] independently. """

    def __call__(self, voxels, **_):
        """ Perform scaling of input. """
        if voxels.ndim == 3:
            voxels = piecewise_linear_warping(voxels, axis=0, symmetric=True)
            voxels = piecewise_linear_warping(voxels, axis=1, symmetric=False)
            voxels = piecewise_linear_warping(voxels, axis=2, symmetric=True)
        else:
            for n in range(voxels.ndim):
                voxels = piecewise_linear_warping(voxels, axis=n)

        return voxels
