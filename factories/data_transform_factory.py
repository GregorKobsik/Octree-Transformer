from data.transform import (
    BasicTransform,
    DoubleConvolutionalTransform,
    TrianryTransform,
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
    if name.startswith("double_conv"):
        return DoubleConvolutionalTransform(architecture)
    elif name.startswith("trinary"):
        return TrianryTransform(architecture, spatial_dim)
    else:
        BasicTransform(architecture)
