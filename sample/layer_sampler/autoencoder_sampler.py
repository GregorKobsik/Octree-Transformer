import torch
from ..sample_utils import preprocess, postprocess


class AutoencoderSampler():
    def __init__(self, model, embedding, head, spatial_dim, max_resolution, position_encoding, device, **_):
        """ Provides a basic implementation of the sampler for the 'autoencoder' architecture.

        Args:
            model: Model which is used for sampling.
            embedding: Token embedding type used in the model.
            head: Generative head type used in the model.
            spatial_dim: The spatial dimensionality of the array of elements.
            max_resolution: Maximum resolution the model is trained on.
            position_encoding: Defines the positional encoding of the data.
            device: Device on which, the data should be stored. Either "cpu" or "cuda" (gpu-support).
        """
        self.model = model
        self.embedding = embedding
        self.head = head
        self.spatial_dim = spatial_dim
        self.max_resolution = max_resolution
        self.pos_encoding = position_encoding
        self.device = device

    def __call__(self, precondition, precondition_resolution, target_resolution, temperature):
        """ Run the model once on the last layer of the input sequence and sample new values for each token.

        Args:
            precondition: An array of elements (pixels/voxels) as an numpy array.
            precondition_resolution: Resolution at which the autoencoder will reconstruct the layer.
            target_resolution: unused.
            temperature: Defines the randomness of the samples.

        Return:
            A token sequence with values, encoding the final sample.
        """
        # transform voxel data into sequences
        val, dep, pos = preprocess(
            precondition, precondition_resolution, self.spatial_dim, self.device, self.pos_encoding
        )
        max_dep = max(dep)

        # process only the last layer of the sequence
        logits = self.model(
            (
                val[dep == max_dep].unsqueeze(0),  # [N, T]
                dep[dep == max_dep].unsqueeze(0),  # [N, T]
                pos[dep == max_dep].unsqueeze(0),  # [N, T, A]
            )
        )[0]  # [T, V]

        # compute token probabilities from logits
        probs = torch.nn.functional.softmax(logits / temperature, dim=1)  # [T, V]

        # zero probability for special tokens -> invalid with parent token
        probs[:, 0] = 0  # 'padding' token

        # sample new value for each token
        new_val = val[dep == max_dep]
        for idx, token_probs in enumerate(probs):
            new_val[idx] = torch.multinomial(token_probs, num_samples=1)[0]

        # replace old values with new ones
        val[dep == max_dep] = new_val

        return postprocess(val, target_resolution, self.spatial_dim)
