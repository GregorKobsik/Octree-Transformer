import torch
from sample.sampler import AbstractSampler


class AutoencoderSampler(AbstractSampler):
    def __init__(self, model, embedding, head, spatial_dim, device, **_):
        """ Provides a basic implementation of the sampler for the autoencoder.

        The following sampler works with the following combinations of modules [architecture, embedding, head]:
            - 'autoencoder', '*', '*'

        Args:
            model: Model which is used for sampling.
            embedding: Token embedding type used in the model.
            head: Generative head type used in the model.
            spatial_dim: The spatial dimensionality of the array of elements.
            device: Device on which, the data should be stored. Either "cpu" or "cuda" (gpu-support).
        """
        super(AutoencoderSampler, self).__init__(model, embedding, head, spatial_dim, device)

    def sample(self, sequences, target_resolution, temperature):
        """ Run the model once on the last layer of the input sequence and sample new values for each token.

        Args:
            sequences: Token sequences, consisting of values, depth and position sequences.
            target_resolution: unused.
            temperatur: Defines the randomness of the samples.

        Return:
            A token sequence with values, encoding the final sample.
        """
        val, dep, pos = sequences
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

        return val
