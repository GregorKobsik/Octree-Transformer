from modules.generative_head import (
    LinearHead,
    SingleConvolutionalHeadA,
    SplitHeadB,
    SubstitutionHead,
)


def create_head(name, num_vocab, embed_dim, spatial_dim):
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
    if name in ('generative_basic', 'linear'):
        return LinearHead(num_vocab, embed_dim)
    elif name in ('single_conv', 'single_conv_A'):
        return SingleConvolutionalHeadA(num_vocab, embed_dim, spatial_dim)
    elif name == 'split_B':
        return SplitHeadB(num_vocab, embed_dim, spatial_dim)
    elif name == 'substitution':
        return SubstitutionHead(num_vocab, embed_dim, spatial_dim)
    elif name == 'discrete_transformation':
        return LinearHead(num_vocab**2**spatial_dim + 1, embed_dim)
    else:
        raise ValueError(f"ERROR: {name} head not implemented.")
