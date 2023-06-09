from .autoencoder import Autoencoder
from .transformer import Transformer
from .pytorch_transformer import PytorchTransformer


def create_architecture(
    architecture,
    attention,
    token_embedding,
    generative_head,
    embed_dim,
    num_heads,
    num_layers,
    dropout,
    num_classes,
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
        dropout: The dropout value.
        num_classes: If bigger, that one the transformer will be class conditional

    Return:
        Transformer model initialised with given parameters.
    """

    kwargs = {
        'token_embedding': token_embedding,
        'generative_head': generative_head,
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'dropout': dropout,
        'num_classes': num_classes
    }

    if architecture == "autoencoder":
        return Autoencoder(**kwargs)
    elif architecture == "encoder_only":
        assert len(token_embedding) == 1, "Only one token embedding allowed."
        assert len(generative_head) == 1, "Only one generative head allowed."
        return Transformer(**kwargs, num_decoders=0)
    elif architecture == "encoder_decoder":
        assert len(token_embedding) == 2, "Only two token embeddings allowed."
        assert len(generative_head) == 1, "Only one generative head allowed."
        generative_head.insert(0, None)
        return Transformer(**kwargs, num_decoders=1)
    elif architecture == "encoder_multi_decoder":
        return Transformer(**kwargs, num_decoders=len(token_embedding) - 1)
    elif architecture == "pytorch":
        return PytorchTransformer(**kwargs)
    elif architecture == "fast":
        # include `pytorch-fast-transformers` as an optional module
        from .fast_transformer import FastTransformer
        return FastTransformer(**kwargs)
    elif architecture == "fast-recurrent":
        # include `pytorch-fast-transformers` as an optional module
        from .fast_recurrent_transformer import FastRecurrentTransformer
        return FastRecurrentTransformer(**kwargs)
    else:
        raise ValueError(f"ERROR: {attention}_{architecture} transformer architecture not implemented.")
