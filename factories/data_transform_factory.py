from data.transform import (
    BasicTransform,
    TrinaryTransform,
)


def create_data_transform(name, spatial_dim):
    """ Creates a data transformation function.

    Args:
        name: Defines which data transformation function will be created.
        spatial_dim: Spatial dimensionality of input data.

    Return:
        data transformation function initialised with specified parameters.
    """
    if name.startswith(('basic', 'single_conv', 'concat', 'half_conv', 'multi_conv')):
        return BasicTransform()
    elif name.startswith('discrete'):
        return TrinaryTransform(spatial_dim)
    else:
        raise ValueError(f"ERROR: No data transform for {name} embedding available.")
