import torch.nn as nn

from .linear_head import LinearHead
from .single_conv_head_A import SingleConvolutionalHeadA
from .split_head_B import SplitHeadB
from .substitution_head import SubstitutionHead
from .half_conv_head_A import HalfConvolutionalHeadA
from .multi_conv_head_A import MultiConvolutionalHeadA


def _create_head(name, num_vocab, embed_dim, spatial_dim):
    """ Creates a generative head.

    If the module specified in `name` does not exist raises a value error.

    Args:
        name: Defines which generative head will be created.
        num_vocab: Number of different vocabs in the vocabulary set.
        embed_dim: Size of embedding dimensions used by the transformer model.
        spatial_dim: Spatial dimensionality of input data.

    Return:
        Generative head initialised with specified parameters.
    """
    if name in ('generative_basic', 'linear', 'basic'):
        return LinearHead(num_vocab, embed_dim)
    elif name in ('single_conv', 'single_conv_A'):
        return SingleConvolutionalHeadA(num_vocab, embed_dim, spatial_dim)
    elif name == 'split_B':
        return SplitHeadB(num_vocab, embed_dim, spatial_dim)
    elif name == 'substitution':
        return SubstitutionHead(num_vocab, embed_dim, spatial_dim)
    elif name == 'discrete_transformation':
        return LinearHead(num_vocab**2**spatial_dim + 1, embed_dim)
    elif name == 'half_conv_A':
        return HalfConvolutionalHeadA(num_vocab, embed_dim, spatial_dim)
    elif name == 'multi_conv_A':
        return MultiConvolutionalHeadA(num_vocab, embed_dim, spatial_dim)
    else:
        raise ValueError(f"ERROR: {name} head not implemented.")


def create_head(name, num_vocab, embed_dim, spatial_dim):
    """ Creates a generative head.

    If `name` is a list, creates a list of heads for each element of the list, otherwise a single one. If the module
    specified in `name` does not exist raises a value error.


    Args:
        name: Defines which generative head will be created.
        num_vocab: Number of different vocabs in the vocabulary set.
        embed_dim: Size of embedding dimensions used by the transformer model.
        spatial_dim: Spatial dimensionality of input data.

    Return:
        Generative head or a list of heads initialised with specified parameters.
    """
    if type(name) == list:
        return nn.ModuleList([_create_head(n, num_vocab, embed_dim, spatial_dim) for n in name])
    else:
        return _create_head(name, num_vocab, embed_dim, spatial_dim)
