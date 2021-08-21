from .cross_entropy_loss import CrossEntropyLoss
from .depth_weighted_cross_entropy_loss_A import DepthWeightedCrossEntropyLossA
from .depth_weighted_cross_entropy_loss_B import DepthWeightedCrossEntropyLossB
from .depth_weighted_cross_entropy_loss_C import DepthWeightedCrossEntropyLossC


def create_loss(name, ignore_index, max_depth, spatial_dim):
    """ Creates a loss function.

    If the module specified in `name` does not exist raises a value error.

    Args:
        name: Defines which loss function will be created.
        ignore_index: Defines an index (padding token), for which the gradient will be not computed.
        max_depth: Maximum depth layer of input data.
        spatial_dim: Spatial dimensionality of input data.

    Return:
        Loss function initialised with specified parameters.
    """
    kwargs = {
        "ignore_index": ignore_index,
        "max_depth": max_depth,
        "spatial_dim": spatial_dim,
    }

    if name == 'cross_entropy':
        return CrossEntropyLoss(**kwargs)
    elif name == 'depth_cross_entropy_A':
        return DepthWeightedCrossEntropyLossA(**kwargs)
    elif name == 'depth_cross_entropy_B':
        return DepthWeightedCrossEntropyLossB(**kwargs)
    elif name == 'depth_cross_entropy_C':
        return DepthWeightedCrossEntropyLossC(**kwargs)
    else:
        raise ValueError(f"ERROR: {name} loss not implemented.")
