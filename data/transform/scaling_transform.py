from utils import axis_scaling


class AxisScalingTransform():
    def __init__(self, **_):
        """Scales input data for each axis in the range of [0.75 .. 1.25] independently. """

    def __call__(self, voxels, **_):
        """ Perform scaling of input. """
        return axis_scaling(voxels)
