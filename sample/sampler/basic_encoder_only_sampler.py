class BasicEncoderOnlySampler():
    def __init__(self, spatial_dim, device, max_tokens, max_resolution, model, **_):
        """ Provides a basic implementation of the sampler for the encoder only architecture with all basic modules.

        The following sampler works with the following combinations of modules [architecture, embedding, head]:
            - 'encoder_only', 'basic', 'generative_basic'

        Args:
            spatial_dim: The spatial dimensionality of the array of elements.
            device: The device on which, the data should be stored. Either "cpu" or "cuda" (gpu-support).
            max_tokens: The maximum number of tokens a sequence can have.
            max_resolution: The maximum resolution the model is trained on
            model: The model which is used for sampling.
        """
        self.spatial_dim = spatial_dim
        self.device = device
        self.max_tokens = max_tokens
        self.max_resolution = max_resolution
        self.model = model

    def __call__(self, precondition, precondition_resolution, target_resolution, temperature):
        """ TODO: add description

        """
        # preprocess the input and transform it into a token sequence
        sequence = self.preprocess(precondition, precondition_resolution)

        # enhance the resolution of the sequence or generate a new sequence by sampling new token values
        value = self.sample(sequence, target_resolution, temperature)

        # postprocess the token value sequence and return it as an array of elements
        return self.postprocess(value, target_resolution)

    def preprocess(self, precondition, precondition_resolution):
        """ Transform input array elements into token sequences.

        Args:
            precondition: An array of elements (pixels/voxels) as an numpy array.
            precondition_resolution: Resolution, to which the input array will be downscaled and used as a precondition
                for sampling.

        Return:
            PyTorch tensor consisting of token sequences: (value, depth, position).
        """
        return 0, 0, 0

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

    def postprocess(self, value, target_resolution):
        """ Transform sequence of value tokens into an array of elements (voxels/pixels).

        Args:
            value: Value token sequence as a pytorch tensor.
            target_resolution: Resolution up to which an object should be sampled.

        Return:
            An array of elements as a numpy array.
        """
        return 0
