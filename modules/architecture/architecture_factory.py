from modules.architecture.autoencoder import Autoencoder
from modules.architecture.encoder_only import EncoderOnly
from modules.architecture.encoder_decoder import EncoderDecoder
from modules.architecture.encoder_multi_decoder import EncoderMultiDecoder


def create_architecture(
    architecture,
    attention,
    token_embedding,
    generative_head,
    embed_dim,
    num_heads,
    num_layers,
    num_positions,
):
    """ Creates a transformer model.

    If the module specified in `name` does not exist raises a value error.

    Args:
        architecture: Defines the underlying architecture of the transformer model.
        attention: Defines the attention of the transformer mode.
        token_embedding: Instance of a token embedding layer.
        generative_head: Instance of a generative head.
        embed_dim: Size of embedding dimensions used by the transformer model.
        num_heads: Number of heads used by the attention.
        num_layers: Number of layers for each of 'decoder' and 'encoder' part in the transformer.
        num_positions: Maximal length of processed input tokens by the attention layers.

    Return:
        Transformer model initialised with given parameters.
    """

    kwargs = {
        'token_embedding': token_embedding,
        'generative_head': generative_head,
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'num_positions': num_positions,
    }

    if architecture == "autoencoder":
        return Autoencoder(**kwargs)
    elif architecture == "encoder_only":
        return EncoderOnly(**kwargs)
    elif architecture == "encoder_decoder":
        return EncoderDecoder(**kwargs)
    elif architecture == "encoder_multi_decoder":
        return EncoderMultiDecoder(**kwargs)
    else:
        raise ValueError(f"ERROR: {attention}_{architecture} shape transformer not implemented.")
