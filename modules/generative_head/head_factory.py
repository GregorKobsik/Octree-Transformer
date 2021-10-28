import torch.nn as nn

from modules.utils import PositionalEncodingLearned, PositionalEncodingLearnedLookAhead, \
    PositionalEncodingLearnedLookAheadSplit
from .composite_head_A import CompositeHeadA, CompositeHeadAutoregressiveA
from .composite_head_B import CompositeHeadB
from .composite_head_C import CompositeHeadC
from .convolution_head_A import ConvolutionHeadA
from .double_substitution_head import DoubleSubstitutionHead
from .linear_head import LinearHead
from .multi_conv_head_A import MultiConvolutionHeadA
from .substitution_head import SubstitutionHead


def _create_head(name, positional_encoding, num_vocab, embed_dim, resolution, spatial_dim):
    """ Creates a generative head.

    If the module specified in `name` does not exist raises a value error.

    Args:
        positional_encoding: positional encoding, that will be added before tokens are generated
        name: Defines which generative head will be created.
        num_vocab: Number of different vocabs in the vocabulary set.
        embed_dim: Size of embedding dimensions used by the transformer model.
        resolution: Spatial resolution of sequence encoding.
        spatial_dim: Spatial dimensionality of input data.

    Return:
        Generative head initialised with specified parameters.
    """

    if positional_encoding == 'None':
        spatial_encoding = None
    elif positional_encoding == 'basic':
        spatial_encoding = PositionalEncodingLearned(embed_dim, resolution, spatial_dim)
    elif positional_encoding == 'look_ahead':
        spatial_encoding = PositionalEncodingLearnedLookAhead(embed_dim, resolution, spatial_dim)
    elif positional_encoding == 'look_ahead_split':
        spatial_encoding = PositionalEncodingLearnedLookAheadSplit(embed_dim, resolution, spatial_dim)
    else:
        raise ValueError(f"ERROR: {positional_encoding} encoding not implemented.")

    kwargs = {
        "spatial_encoding": spatial_encoding,
        "num_vocab": num_vocab,
        "embed_dim": embed_dim,
        "resolution": resolution,
        "spatial_dim": spatial_dim,
        "conv_size": 2 ** spatial_dim,
    }

    if name in ('generative_basic', 'linear', 'basic'):
        return LinearHead(**kwargs)
    elif name == 'discrete_transformation':
        kwargs["num_vocab"] = num_vocab ** 2 ** spatial_dim
        return LinearHead(**kwargs)
    elif name in ('half_conv', 'half_conv_A'):
        kwargs["conv_size"] = 2 ** (spatial_dim - 1)
        return ConvolutionHeadA(**kwargs)
    elif name in ('single_conv', 'single_conv_A'):
        return ConvolutionHeadA(**kwargs)
    elif name == 'multi_conv_A':
        return MultiConvolutionHeadA(**kwargs)
    elif name == 'substitution':
        return SubstitutionHead(**kwargs)
    elif name == 'double_substitution':
        return DoubleSubstitutionHead(**kwargs)
    elif name in ('composite', 'composite_A'):
        return CompositeHeadA(**kwargs)
    elif name in ('composite_autoregressive_A'):
        return CompositeHeadAutoregressiveA(**kwargs)
    elif name in ('composite_B'):
        return CompositeHeadB(**kwargs)
    elif name in ('composite_C'):
        return CompositeHeadC(**kwargs)
    else:
        raise ValueError(f"ERROR: {name} head not implemented.")


def create_head(name, positional_encoding, num_vocab, embed_dim, resolution, spatial_dim):
    """ Creates a generative head.

    If `name` is a list, creates a list of heads for each element of the list, otherwise a single one. If the module
    specified in `name` does not exist raises a value error.


    Args:
        positional_encoding: positional encoding, that will be added before tokens are generated
        name: Defines which generative head will be created.
        num_vocab: Number of different vocabs in the vocabulary set.
        embed_dim: Size of embedding dimensions used by the transformer model.
        resolution: Spatial resolution of sequence encoding.
        spatial_dim: Spatial dimensionality of input data.

    Return:
        Generative head or a list of heads initialised with specified parameters.
    """
    if type(name) == list:
        return nn.ModuleList(
            [_create_head(n, positional_encoding, num_vocab, embed_dim, resolution, spatial_dim) for n in name])
    else:
        return _create_head(name, positional_encoding, num_vocab, embed_dim, resolution, spatial_dim)
