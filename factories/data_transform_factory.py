from data.transform import (
    BasicTransform,
    TrinaryTransform,
    PenultimateLayerTransform,
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
    if name.startswith(('basic', 'single_conv', 'concat', 'half_conv', 'multi_conv')):
        return BasicTransform(architecture)
    elif name.startswith('discrete'):
        return TrinaryTransform(architecture, spatial_dim)
    elif name.startswith('substitution'):
        return PenultimateLayerTransform(architecture)
    else:
        raise ValueError(f"ERROR: No data transform for {name} embedding available.")
