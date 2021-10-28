import math

import torch
from tqdm.auto import tqdm

from ..sample_utils import (
    next_layer_tokens,
    preprocess,
    postprocess,
)
from ..token_generator import create_token_generator


class EncoderOnlySampler:
    def __init__(self, model, head, spatial_dim, max_resolution, position_encoding, device, **_):
        """ Provides a basic implementation of the sampler for the 'encoder_only' architecture.

        Args:
            model: Model which is used for sampling.
            head: Generative head type used in the model.
            spatial_dim: The spatial dimensionality of the array of elements.
            max_resolution: Maximum resolution the model is trained on.
            position_encoding: Defines the positional encoding of the data.
            device: Device on which, the data should be stored. Either "cpu" or "cuda" (gpu-support).
        """
        self.generators = create_token_generator(head, model, spatial_dim)

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

            # sample layer-wise
            for _ in tqdm(range(cur_layer, max_layer), initial=cur_layer, total=max_layer, leave=True, desc="Layers"):

                # init sequences for next layer
                next_val, next_dep, next_pos = next_layer_tokens(
                    val, dep, pos, self.spatial_dim, self.max_resolution, self.pos_encoding
                )
                # predict value tokens for current layer
                next_val = self.generators[0](
                    val=val + [next_val],
                    dep=dep + [next_dep],
                    pos=pos + [next_pos],
                    memory=None,
                    temperature=temperature,
                    cls=cls
                )

                # append sampled tokens to current sequence
                val += [next_val]
                dep += [next_dep]
                pos += [next_pos]

                if len(next_val) != len(next_dep):
                    break  # reached maximum number of tokens which can be generated

        return postprocess(val, target_resolution, self.spatial_dim)
