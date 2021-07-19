from .basic_generator import BasicGenerator
from .composite_generator import CompositeGenerator


def _create_token_generator(head, model, spatial_dim):
    """ Creates a token generator.

    If the module specified in `head` does not exist raises a value error.

    Args:
        head: Generative head type used in `model`.
        model: Model which is used for sampling.
        spatial_dim: Spatial dimensionality of input data.

    Return:
        Token generator initialised with specified parameters.
    """
    if head in ('generative_basic', 'basic', 'linear'):
        return BasicGenerator(model.compute_logits)
    if head in ('half_conv', 'halv_conv_A'):
        return BasicGenerator(model.compute_logits, 2**(spatial_dim - 1))
    if head in ('single_conv', 'single_conv_A'):
        return BasicGenerator(model.compute_logits, 2**spatial_dim)
    if head in ('composite', 'composite_A'):
        return CompositeGenerator(
            model.compute_logits, [1, 1, 1, 2**(spatial_dim - 1), 2**spatial_dim, (2**spatial_dim)**2]
        )
    raise ValueError(f"ERROR: {head} token generator not implemented.")


def create_token_generator(head, model, spatial_dim):
    """ Creates a token generator or a list of token generators.

    If `head` is a list, creates a list of embeddings for each element of the list, otherwise a single one. If the
    module specified in `head` does not exist raises a value error.

    Args:
        head: Generative head type used in `model`.
        model: Model which is used for sampling.
        spatial_dim: Spatial dimensionality of input data.

    Return:
        Token generator or a list of generators initialised with specified parameters.
    """
    if type(head) == list:
        return [_create_token_generator(n, model, spatial_dim) for n in head]
    else:
        return _create_token_generator(head, model, spatial_dim)
