from .basic_generator import BasicGenerator
from .substitution_generator import SubstitutionGenerator
from .double_substitution_generator import DoubleSubstitutionGenerator
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
    if head in ('half_conv', 'half_conv_A'):
        return BasicGenerator(model.compute_logits, 2**(spatial_dim - 1))
    if head in ('single_conv', 'single_conv_A'):
        return BasicGenerator(model.compute_logits, 2**spatial_dim)
    if head in ('substitution'):
        return SubstitutionGenerator(model.compute_logits, 2**spatial_dim)
    if head in ('double_substitution'):
        return DoubleSubstitutionGenerator(model.compute_logits, 2**spatial_dim)
    if head in ('composite', 'composite_A'):
        size = 2**spatial_dim
        return CompositeGenerator(model.compute_logits, [1, 1, 1, size // 2, size, size, size])
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
