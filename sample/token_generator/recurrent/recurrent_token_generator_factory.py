from .recurrent_basic_generator import RecurrentBasicGenerator
from .recurrent_composite_generator import RecurrentCompositeGenerator, RecurrentCompositeGeneratorAutoregressive


def _create_recurrent_token_generator(head, model, spatial_dim):
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
        'embed_fn': model.token_embedding,
        'transformer_fn': model.transformer_module,
        'head_fn': model.generative_head,
    }

    if head in ('generative_basic', 'basic', 'linear'):
        return RecurrentBasicGenerator(num_tokens=1, **kwargs)
    if head in ('half_conv', 'half_conv_A'):
        return RecurrentBasicGenerator(num_tokens=4, **kwargs)
    if head in ('single_conv', 'single_conv_A'):
        return RecurrentBasicGenerator(num_tokens=8, **kwargs)
    # if head in ('substitution'):
    #     return RecurrentSubstitutionGenerator(model.compute_logits, 2**spatial_dim)
    # if head in ('double_substitution'):
    #     return RecurrentDoubleSubstitutionGenerator(model.compute_logits, 2**spatial_dim)
    if head in ('composite', 'composite_A', 'composite_B'):
        return RecurrentCompositeGenerator(num_tokens=[1, 1, 1, 4, 8, 8, 8, 8], **kwargs)
    if head in ('composite_autoregressive_A'):
        return RecurrentCompositeGeneratorAutoregressive(num_tokens=[1, 1, 1, 4, 8, 8, 8, 8], **kwargs)
    # if head in ('composite_B'):
    #     size = 2**spatial_dim
    #     return RecurrentCompositeGenerator(model.compute_logits, [1, 1, 1, size // 4, size, size, size, size])
    raise ValueError(f"ERROR: {head} token generator not implemented.")


def create_recurrent_token_generator(head, model, spatial_dim):
    """ Creates a recurrent token generator or a list of token generators.

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
        return [_create_recurrent_token_generator(n, model, spatial_dim) for n in head]
    else:
        return _create_recurrent_token_generator(head, model, spatial_dim)
