from sample.sampler import AbstractSampler


class BasicEncoderOnlySampler(AbstractSampler):
    def __init__(self, model, embedding, head, spatial_dim, max_tokens, max_resolution, device, **_):
        """ Provides a basic implementation of the sampler for the encoder only architecture with all basic modules.

        The following sampler works with the following combinations of modules [architecture, embedding, head]:
            - 'encoder_only', 'basic', 'generative_basic'

        Args:
            model: Model which is used for sampling.
            embedding: Token embedding type used in the model.
            head: Generative head type used in the model.
            spatial_dim: Spatial dimensionality of the array of elements.
            device: Device on which, the data should be stored. Either "cpu" or "cuda" (gpu-support).
            max_tokens: Maximum number of tokens a sequence can have.
            max_resolution: Maximum resolution the model is trained on.
        """
        super(BasicEncoderOnlySampler, self).__init__(model, embedding, head, spatial_dim, device)
        self.max_tokens = max_tokens
        self.max_resolution = max_resolution

    def sample(self, sequences, target_resolution, temperature):
        """ Perform an iterative sampling of the given sequence until reaching the end of sequence, the maximum sequence
            length or the desired resolution.

        Args:
            sequences: Token sequences, consisting of values, depth and position sequences.
            target_resolution: Resolution up to which an object should be sampled.
            temperatur: Defines the randomness of the samples.

        Return:
            A token sequence with values, encoding the final sample.
        """
        return 0
