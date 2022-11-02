from .autoencoder_collate import AutoencoderCollate
from .encoder_only_collate import EncoderOnlyCollate
from .encoder_decoder_collate import EncoderDecoderCollate
from .encoder_multi_decoder_collate import EncoderMultiDecoderCollate


def create_data_collate(architecture, embeddings, resolution):
    """ Creates a data collate function.

    Args:
        architecture: Transformer architecture defines which data transformation function will be created.
        embeddings: Defines the used token embeddings in the shape transformer.
        resolution: Maximum side length of input data.

    Return:
        data transformation function initialised with specified parameters.
    """
    if architecture == "autoencoder":
        return AutoencoderCollate(embeddings)
    if architecture in ("encoder_only", 'pytorch', 'fast', 'fast-recurrent', 'fast_recurrent', 'sliding_window'):
        return EncoderOnlyCollate()
    if architecture == "encoder_decoder":
        return EncoderDecoderCollate(embeddings)
    if architecture == "encoder_multi_decoder":
        return EncoderMultiDecoderCollate(embeddings, resolution)
    else:
        raise ValueError(f"ERROR: No data collate for {architecture} available.")
