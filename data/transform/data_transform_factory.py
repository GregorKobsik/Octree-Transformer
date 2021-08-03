from .basic_transform import BasicTransform
from .trinary_transform import TrinaryTransform


def create_data_transform(name, spatial_dim, resolution, position_encoding):
    """ Creates a data transformation function.

    Args:
        name: Defines which data transformation function will be created.
        spatial_dim: Spatial dimensionality of input data.

    Return:
        data transformation function initialised with specified parameters.
    """
    if type(name) == list:
        kwargs = {
            "spatial_dim": 3,
            "resolution": resolution,
            "position_encoding": position_encoding,
        }
        return BasicTransform(**kwargs)
    elif name.startswith(('basic', 'single_conv', 'concat', 'half_conv', 'multi_conv')):
        return BasicTransform(**kwargs)
    elif name.startswith('discrete'):
        return TrinaryTransform(spatial_dim)
    else:
        raise ValueError(f"ERROR: No data transform for {name} embedding available.")
