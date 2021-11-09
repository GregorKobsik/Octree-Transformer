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
    kwargs = {
        'compute_logits_fn': model.compute_logits,
    }

    if head in ('composite_A'):
        return CompositeGenerator(num_tokens=[1, 1, 1, 4, 8, 8, 8, 8], **kwargs)
    if head in ('composite_B'):
        return CompositeGenerator(num_tokens=[1, 1, 1, 1, 8, 8], **kwargs)
    if head in ('composite_C'):
        return CompositeGenerator(num_tokens=[1, 1, 2, 4, 8, 4], **kwargs)
    if head in ('composite_D'):
        return CompositeGenerator(num_tokens=[1, 1, 4, 8, 4, 8, 4, 8], **kwargs)
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
