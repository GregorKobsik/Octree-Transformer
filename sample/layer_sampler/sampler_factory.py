from .autoencoder_sampler import AutoencoderSampler
from .encoder_only_sampler import EncoderOnlySampler
from .encoder_decoder_sampler import EncoderDecoderSampler
from .encoder_multi_decoder_sampler import EncoderMultiDecoderSampler
from .recurrent_sampler import RecurrentSampler


def create_sampler(
    architecture, embedding, head, model, spatial_dim, max_tokens, max_resolution, position_encoding, device
):
    """ Creates a sampler model.

    If the module does not exist raises a value error.

    Args:
        architecture: Architecture type used in `model`.
        embedding: Token embedding type used in `model`.
        head: Generative head type used in `model`.
        model: Model which is used for sampling.
        spatial_dim: Spatial dimensionality of the array of elements.
        device: Device on which, the data should be stored. Either "cpu" or "cuda" (gpu-support).
        max_tokens: Maximum number of tokens a sequence can have.
        max_resolution: Maximum resolution the model is trained on.
        position_encoding: Defines the positional encoding of the data.
    """

    kwargs = {
        "model": model,
        "embedding": embedding,
        "head": head,
        "spatial_dim": spatial_dim,
        "device": device,
        "max_tokens": max_tokens,
        "max_resolution": max_resolution,
        "position_encoding": position_encoding,
    }

    if architecture == "autoencoder":
        return AutoencoderSampler(**kwargs)
    elif architecture in ("encoder_only", "fast", "pytorch"):
        return EncoderOnlySampler(**kwargs)
    elif architecture in ("fast-recurrent"):
        return RecurrentSampler(**kwargs)
    elif architecture == "encoder_decoder":
        return EncoderDecoderSampler(**kwargs)
    elif architecture == "encoder_multi_decoder":
        return EncoderMultiDecoderSampler(**kwargs)
    else:
        raise ValueError(f"No sampler defined for {architecture} architecture.")
