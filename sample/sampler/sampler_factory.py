from sample.sampler.abstract_sampler import AbstractSampler
from sample.sampler.basic_encoder_decoder_sampler import BasicEncoderDecoderSampler
from sample.sampler.single_conv_encoder_decoder_sampler import SingleConvEncoderDecoderSampler
from sample.sampler.double_convolutional_encoder_decoder_sampler import DoubleConvolutionalEncoderDecoderSampler
from sample.sampler.autoencoder_sampler import AutoencoderSampler


def create_sampler(architecture, embedding, head, model, spatial_dim, max_tokens, max_resolution, device):
    """ Creates a sampler model.

    If the module based on architecture, embedding and head exist raises a value error.

    Args:
        architecture: Architecture type used in `model`.
        embedding: Token embedding type used in `model`.
        head: Generative head type used in `model`.
        model: Model which is used for sampling.
        spatial_dim: Spatial dimensionality of the array of elements.
        device: Device on which, the data should be stored. Either "cpu" or "cuda" (gpu-support).
        max_tokens: Maximum number of tokens a sequence can have.
        max_resolution: Maximum resolution the model is trained on.
    """

    kwargs = {
        "model": model,
        "embedding": embedding,
        "head": head,
        "spatial_dim": spatial_dim,
        "device": device,
        "max_tokens": max_tokens,
        "max_resolution": max_resolution,
    }

    if (
        architecture == "encoder_decoder" and embedding.startswith(("basic", "single_conv_F")) and
        head.startswith(("linear", "generative_basic"))
    ):
        return BasicEncoderDecoderSampler(**kwargs)
    elif (
        architecture == "encoder_decoder" and embedding.startswith(("single_conv", "concat")) and
        head.startswith(("single_conv", "split"))
    ):
        return SingleConvEncoderDecoderSampler(**kwargs)
    elif (architecture == "encoder_decoder" and embedding.startswith("double_conv") and head.startswith("double_conv")):
        return DoubleConvolutionalEncoderDecoderSampler(**kwargs)
    elif architecture == "autoencoder":
        return AutoencoderSampler(**kwargs)
    else:
        raise ValueError(
            "No sampler defined for the combination or parameters - " +
            f"architecture: {architecture}, embedding: {embedding}, and head: {head}."
        )
