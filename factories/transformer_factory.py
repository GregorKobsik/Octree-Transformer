from modules.transformer import (
    BasicTransformer,
)


def create_transformer(
    name,
    token_embedding,
    generative_head,
    architecture,
    attention,
    embed_dim,
    num_heads,
    num_layers,
    num_positions,
    num_vocab,
    resolution,
    spatial_dim,
):
    """ Creates a transformer model.

    If the module specified in `name` does not exist raises a value error.

    Args:
        name: Defines which transformer model will be created, e.g. which attention will be used.
        token_embedding: Instance of a token embedding layer.
        generative_head: Instance of a generative head.
        architecture: Defines whether the transformer uses a 'encoder_only' or 'encoder_decocer' architecture.
        embed_dim: Size of embedding dimensions used by the transformer model.
        num_heads: Number of heads used by the attention.
        num_layers: Number of layers for each the 'decoder' and 'encoder' part of the transformer.
        num_positions: Maximal length of processed input tokens by the attention layers.

    Return:
        Transformer model initialised with given parameters.
    """

    kwargs = {
        'token_embedding': token_embedding,
        'generative_head': generative_head,
        'architecture': architecture,
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'num_positions': num_positions,
    }

    if name == "basic":
        return BasicTransformer(**kwargs)
    else:
        raise ValueError(f"ERROR: {name} attention transformer not implemented.")
