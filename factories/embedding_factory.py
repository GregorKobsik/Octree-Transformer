from modules.embedding import (
    BasicEmbedding,
    SingleConvolutionalEmbeddingA,
    SingleConvolutionalEmbeddingB,
    SingleConvolutionalEmbeddingC,
    SingleConvolutionalEmbeddingD,
    SingleConvolutionalEmbeddingE,
    SingleConvolutionalEmbeddingF,
    SingleConvolutionalEmbeddingG,
    SingleConvolutionalEmbeddingH,
    SingleConvolutionalEmbeddingI,
    ConcatEmbeddingA,
    ConcatEmbeddingB,
    ConcatEmbeddingC,
    DoubleConvolutionalEmbedding,
)


def create_embedding(name, num_vocab, embed_dim, resolution, spatial_dim):
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
    if name == 'basic':
        return BasicEmbedding(num_vocab, embed_dim, resolution, spatial_dim)
    elif name in ('single_conv', 'single_conv_A'):
        return SingleConvolutionalEmbeddingA(num_vocab, embed_dim, resolution, spatial_dim)
    elif name == 'single_conv_B':
        return SingleConvolutionalEmbeddingB(embed_dim, spatial_dim)
    elif name == 'single_conv_C':
        return SingleConvolutionalEmbeddingC(num_vocab, embed_dim, resolution, spatial_dim)
    elif name == 'single_conv_D':
        return SingleConvolutionalEmbeddingD(num_vocab, embed_dim, resolution, spatial_dim)
    elif name == 'single_conv_E':
        return SingleConvolutionalEmbeddingE(num_vocab, embed_dim, resolution, spatial_dim)
    elif name == 'single_conv_F':
        return SingleConvolutionalEmbeddingF(num_vocab, embed_dim, resolution, spatial_dim)
    elif name == 'single_conv_G':
        return SingleConvolutionalEmbeddingG(num_vocab, embed_dim, resolution, spatial_dim)
    elif name == 'single_conv_H':
        return SingleConvolutionalEmbeddingH(num_vocab, embed_dim, resolution, spatial_dim)
    elif name == 'single_conv_I':
        return SingleConvolutionalEmbeddingI(num_vocab, embed_dim, resolution, spatial_dim)
    elif name == 'concat_A':
        return ConcatEmbeddingA(num_vocab, embed_dim, resolution, spatial_dim)
    elif name == 'concat_B':
        return ConcatEmbeddingB(num_vocab, embed_dim, resolution, spatial_dim)
    elif name == 'concat_C':
        return ConcatEmbeddingC(num_vocab, embed_dim, resolution, spatial_dim)
    elif name == 'double_conv':
        return DoubleConvolutionalEmbedding(embed_dim, spatial_dim)
    else:
        raise ValueError(f"ERROR: {name} embedding not implemented.")
