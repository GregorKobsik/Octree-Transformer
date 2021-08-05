from .basic_transform import BasicTransform
from .comosite_transform import CompositeTransform
from .scaling_transform import AxisScalingTransform
from .piecewise_warping_transform import PiecewiseLinearWarpingTransform
from .quick_linearisation_transform import QuickLinearisationTransform


def _create_data_transform(name, spatial_dim, resolution, position_encoding):
    """ Creates a data transformation function.

    Args:
        name: Defines which data transformation function will be created.
        spatial_dim: Spatial dimensionality of input data.

    Return:
        data transformation function initialised with specified parameters.
    """
    if name == 'basic':
        return BasicTransform(position_encoding, resolution)
    elif name == 'scaling':
        return AxisScalingTransform()
    elif name == 'warping':
        return PiecewiseLinearWarpingTransform()
    elif name in ('linear_max_res', 'quick_linear', 'linear'):
        return QuickLinearisationTransform(position_encoding, resolution)
    elif name == 'linear_max_16':
        return QuickLinearisationTransform(position_encoding, 16)
    elif name == 'linear_max_32':
        return QuickLinearisationTransform(position_encoding, 32)
    elif name == 'linear_max_64':
        return QuickLinearisationTransform(position_encoding, 64)
    elif name == 'linear_max_128':
        return QuickLinearisationTransform(position_encoding, 128)
    elif name == 'linear_max_256':
        return QuickLinearisationTransform(position_encoding, 256)
    else:
        raise ValueError(f"ERROR: No data transform for {name} available.")


def create_data_transform(name, spatial_dim, resolution, position_encoding):
    """ Creates a data transformation function.

    Args:
        name: Defines which data transformation function will be created.
        spatial_dim: Spatial dimensionality of input data.

    Return:
        data transformation function initialised with specified parameters.
    """
    if type(name) == list:
        return CompositeTransform([_create_data_transform(x, spatial_dim, resolution, position_encoding) for x in name])
    else:
        return _create_data_transform(name, spatial_dim, resolution, position_encoding)
