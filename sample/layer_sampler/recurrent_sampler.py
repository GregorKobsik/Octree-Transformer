import math

import torch
from tqdm.auto import tqdm

from ..sample_utils import (
    next_layer_tokens,
    preprocess,
    postprocess,
)
from ..token_generator.recurrent import create_recurrent_token_generator


class RecurrentSampler:
    def __init__(self, model, head, spatial_dim, max_resolution, position_encoding, device, **_):
        """ Provides a basic implementation of the sampler for the 'fast-recurrent-transformer' architecture.

        Args:
            model: Model which is used for sampling.
            head: Generative head type used in the model.
            spatial_dim: The spatial dimensionality of the array of elements.
            max_resolution: Maximum resolution the model is trained on.
            position_encoding: Defines the positional encoding of the data.
            device: Device on which, the data should be stored. Either "cpu" or "cuda" (gpu-support).
        """
        super(RecurrentSampler, self).__init__()
        self.generators = create_recurrent_token_generator(head, model, spatial_dim)
        self.model = model

        self.spatial_dim = spatial_dim
        self.max_resolution = max_resolution
        self.pos_encoding = position_encoding
        self.device = device

    def __call__(self, precondition, precondition_resolution, target_resolution, temperature, cls):
        """ Perform an iterative sampling of the given sequence until reaching the end of sequence, the maximum sequence
            length or the desired resolution.

        Args:
            precondition: An array of elements (pixels/voxels) as an numpy array.
            precondition_resolution: Resolution at which the autoencoder will reconstruct the layer.
            target_resolution: Resolution up to which an object should be sampled.
            temperature: Defines the randomness of the samples.
            cls: class label for conditional generation.

        Return:
            A token sequence with values, encoding the final sample.
        """
        # transform voxel data into sequences
        val, dep, pos = preprocess(
            precondition, precondition_resolution, self.spatial_dim, self.pos_encoding, self.device
        )

        # compute the number of finished (current) layers and the maximum sampleable layer
        cur_layer = len(val)
        max_layer = int(math.log2(min(target_resolution, self.max_resolution)))

        with torch.no_grad():

            # initialise memory and state of the transformer model for already predefined tokens
            state = None
            seq = (torch.cat(val).unsqueeze(0), torch.cat(dep).unsqueeze(0), torch.cat(pos).unsqueeze(0))
            input_seq = self.model.token_embedding(seq, cls)  # [N, L, E]
            memory = torch.zeros_like(input_seq)

            for i in tqdm(range(input_seq.shape[1]), desc="Initialize"):
                memory[:, i], state = self.model.transformer_module(input_seq[:, i], state=state)

            # sample new tokens layer-wise, as each layer might use a different token embedding and generative head
            for _ in tqdm(range(cur_layer, max_layer), initial=cur_layer, total=max_layer, leave=True, desc="Layers"):

                # init sequences for the current layer based on the previous one
                next_val, next_dep, next_pos = next_layer_tokens(
                    val, dep, pos, self.spatial_dim, self.max_resolution, self.pos_encoding
                )
                # generate value tokens for the current layer
                next_val, memory, state = self.generators[0](
                    val=val + [next_val],
                    dep=dep + [next_dep],
                    pos=pos + [next_pos],
                    memory=memory,
                    state=state,
                    temperature=temperature,
                    cls=cls
                )

                # append sampled tokens to previous sequences
                val += [next_val]
                dep += [next_dep]
                pos += [next_pos]

        # transform the sampled octree sequence back into a regular-grid voxel array and return
        return postprocess(val, target_resolution, self.spatial_dim)
