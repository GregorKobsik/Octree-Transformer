from .basic_transform import BasicTransform
from .comosite_transform import CompositeTransform
from .scaling_transform import AxisScalingTransform
from .piecewise_warping_transform import PiecewiseLinearWarpingTransform
from .quick_linearisation_transform import QuickLinearisationTransform
from .check_sequence_length_transform import CheckSequenceLenghtTransform


def _create_data_transform(name, spatial_dim, resolution, position_encoding, num_positions, embedding):
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
    elif name == 'check_len':
        return CheckSequenceLenghtTransform(num_positions, embedding)
    else:
        raise ValueError(f"ERROR: No data transform for {name} available.")


def create_data_transform(name, spatial_dim, resolution, position_encoding, num_positions, embedding):
    """ Creates a data transformation function.

    Args:
        name: Defines which data transformation function will be created.
        spatial_dim: Spatial dimensionality of input data.

    Return:
        data transformation function initialised with specified parameters.
    """
    kwargs = {
        'spatial_dim': spatial_dim,
        'resolution': resolution,
        'position_encoding': position_encoding,
        'num_positions': num_positions,
        'embedding': embedding,
    }

    if name is None:
        return None
    elif type(name) == list:
        return CompositeTransform([_create_data_transform(x, **kwargs) for x in name])
    else:
        return _create_data_transform(name, **kwargs)
