import torch.nn as nn

from modules.embedding import (
    BasicEmbeddingA,
    HalfConvolutionalEmbeddingA,
    SingleConvolutionalEmbeddingA,
    MultiConvolutionalEmbeddingA,
    SubstitutionEmbedding,
)


def _create_embedding(name, num_vocab, embed_dim, resolution, spatial_dim):
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
    if name in ('basic', 'basic_A'):
        return BasicEmbeddingA(num_vocab, embed_dim, resolution, spatial_dim)
    elif name == 'discrete_transformation':
        return BasicEmbeddingA(num_vocab**2**spatial_dim + 1, embed_dim, resolution, spatial_dim)
    elif name == 'half_conv_A':
        return HalfConvolutionalEmbeddingA(num_vocab, embed_dim, resolution, spatial_dim)
    elif name in ('single_conv', 'single_conv_A'):
        return SingleConvolutionalEmbeddingA(num_vocab, embed_dim, resolution, spatial_dim)
    elif name == 'multi_conv_A':
        return MultiConvolutionalEmbeddingA(num_vocab, embed_dim, resolution, spatial_dim)
    elif name == 'substitution':
        return SubstitutionEmbedding(num_vocab, embed_dim, resolution, spatial_dim)
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
