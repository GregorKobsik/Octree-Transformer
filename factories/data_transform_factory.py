from data.transform import (
    BasicTransform,
    TrinaryTransform,
    DoubleConvolutionalTransform,
)


def create_data_transform(name, architecture, spatial_dim):
    """ Creates a data transformation function.

    Args:
        name: Defines which data transformation function will be created.
        architecture: Defines which architecture is used in the transformer model.
        spatial_dim: Spatial dimensionality of input data.

    Return:
        data transformation function initialised with specified parameters.
    """
    if name.startswith(('basic', 'single_conv', 'concat')):
        return BasicTransform(architecture)
    elif name.startswith('discrete'):
        return TrinaryTransform(architecture, spatial_dim)
    elif name.startswith('double_conv'):
        return DoubleConvolutionalTransform(architecture)
    else:
        raise ValueError(f"ERROR: No data transform for {name} embedding available.")
