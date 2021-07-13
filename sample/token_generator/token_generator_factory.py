from .basic_generator import BasicGenerator


def _create_token_generator(head, model):
    """ Creates a token generator.

    If the module specified in `head` does not exist raises a value error.

    Args:
        head: Generative head type used in `model`.
        model: Model which is used for sampling.

    Return:
        Token generator initialised with specified parameters.
    """
    if head == "basic":
        return BasicGenerator(model.compute_logits)
    raise ValueError(f"ERROR: {head} token generator not implemented.")


def create_token_generator(head, model):
    """ Creates a token generator or a list of token generators.

    If `head` is a list, creates a list of embeddings for each element of the list, otherwise a single one. If the
    module specified in `head` does not exist raises a value error.

    Args:
        head: Generative head type used in `model`.
        model: Model which is used for sampling.

    Return:
        Token generator or a list of generators initialised with specified parameters.
    """
    if type(head) == list:
        return [_create_token_generator(n, model) for n in head]
    else:
        return _create_token_generator(head, model)
