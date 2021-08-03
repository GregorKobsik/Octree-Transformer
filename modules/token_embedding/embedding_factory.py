import torch.nn as nn

from .basic_embedding_A import BasicEmbeddingA
from .convolution_embedding_A import ConvolutionEmbeddingA
from .multi_conv_embedding_A import MultiConvolutionEmbeddingA
from .substitution_embedding import SubstitutionEmbedding
from .double_substitution_embedding import DoubleSubstitutionEmbedding
from .composite_embedding_A import CompositeEmbeddingA
from .composite_embedding_B import CompositeEmbeddingB


def _create_embedding(name, num_vocab, embed_dim, resolution, spatial_dim, **_):
    """ Creates a token embedding.

    If the module specified in `name` does not exist raises a value error.

    Args:
        name: Defines which token embedding will be created.
        num_vocab: Number of different vocabs in the vocabulary set.
        embed_dim: Size of embedding dimensions used by the transformer model.
        resolution: Maximum side length of input data.
        spatial_dim: Spatial dimensionality of input data.

    Return:
        Token embedding initialised with specified parameters.
    """
    kwargs = {
        "num_vocab": num_vocab,
        "embed_dim": embed_dim,
        "resolution": resolution,
        "spatial_dim": spatial_dim,
        "conv_size": 2**spatial_dim,
    }

    if name in ('basic', 'basic_A'):
        return BasicEmbeddingA(**kwargs)
    elif name == 'discrete_transformation':
        kwargs['num_vocab'] = num_vocab**2**spatial_dim
        return BasicEmbeddingA(**kwargs)
    elif name in ('half_conv', 'half_conv_A'):
        kwargs['conv_size'] = 2**(spatial_dim - 1)
        return ConvolutionEmbeddingA(**kwargs)
    elif name in ('single_conv', 'single_conv_A'):
        return ConvolutionEmbeddingA(**kwargs)
    elif name == 'multi_conv_A':
        return MultiConvolutionEmbeddingA(**kwargs)
    elif name == 'substitution':
        return SubstitutionEmbedding(**kwargs)
    elif name == 'double_substitution':
        return DoubleSubstitutionEmbedding(**kwargs)
    elif name in ('composite', 'composite_A'):
        return CompositeEmbeddingA(**kwargs)
    elif name in ('composite_B'):
        return CompositeEmbeddingB(**kwargs)
    else:
        raise ValueError(f"ERROR: {name} embedding not implemented.")


def create_embedding(name, num_vocab, embed_dim, resolution, spatial_dim):
    """ Creates a token embedding.

    If `name` is a list, creates a list of embeddings for each element of the list, otherwise a single one. If the
    module specified in `name` does not exist raises a value error.

    Args:
        name: Defines which token embedding will be created.
        num_vocab: Number of different vocabs in the vocabulary set.
        embed_dim: Size of embedding dimensions used by the transformer model.
        resolution: Maximum side length of input data.
        spatial_dim: Spatial dimensionality of input data.

    Return:
        Token embedding or a list of embeddings initialised with specified parameters.
    """
    if type(name) == list:
        return nn.ModuleList([_create_embedding(n, num_vocab, embed_dim, resolution, spatial_dim) for n in name])
    else:
        return _create_embedding(name, num_vocab, embed_dim, resolution, spatial_dim)
