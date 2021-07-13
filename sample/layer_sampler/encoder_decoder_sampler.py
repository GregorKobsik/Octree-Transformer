import torch
import math
from tqdm.auto import tqdm

from ..token_generator import create_token_generator
from ..sample_utils import (
    next_layer_tokens,
    preprocess,
    postprocess,
)


class EncoderDecoderSampler():
    def __init__(self, model, head, spatial_dim, max_resolution, device, **_):
        """ Provides a basic implementation of the sampler for the 'encoder_only' architecture.

        Args:
            model: Model which is used for sampling.
            head: Generative head type used in the model.
            spatial_dim: The spatial dimensionality of the array of elements.
            max_resolution: Maximum resolution the model is trained on.
            device: Device on which, the data should be stored. Either "cpu" or "cuda" (gpu-support).
        """
        self.generators = create_token_generator(head, model)
        self.compute_memory = model.compute_memory

        self.spatial_dim = spatial_dim
        self.max_resolution = max_resolution
        self.device = device

    def __call__(self, precondition, precondition_resolution, target_resolution, temperature):
        """ Perform an iterative sampling of the given sequence until reaching the end of sequence, the maximum sequence
            length or the desired resolution.

        Args:
            precondition: An array of elements (pixels/voxels) as an numpy array.
            precondition_resolution: Resolution at which the autoencoder will reconstruct the layer.
            target_resolution: Resolution up to which an object should be sampled.
            temperature: Defines the randomness of the samples.

        Return:
            A token sequence with values, encoding the final sample.
        """
        # transform voxel data into sequences
        val, dep, pos = preprocess(precondition, precondition_resolution, self.spatial_dim, self.device)

        # compute the number of finished (current) layers and the maximum sampleable layer
        cur_layer = 0 if len(dep) == 0 else int(max(dep))
        max_layer = int(math.log2(min(target_resolution, self.max_resolution)))

        with torch.no_grad():

            # sample layer-wise
            for _ in tqdm(range(cur_layer, max_layer), initial=cur_layer, total=max_layer, leave=True, desc="Layers"):

                # compute memory / encode sequence
                seq = (
                    val.unsqueeze(0),
                    dep.unsqueeze(0),
                    pos.unsqueeze(0),
                )
                memory = self.compute_memory(seq, memory=None, idx=0, is_final=False)

                # init sequences for next layer
                layer_val, layer_dep, layer_pos = next_layer_tokens(
                    val, dep, pos, self.spatial_dim, self.max_resolution
                )

                # predict value tokens for current layer / decode sequence
                layer_val = self.generators[0](  # TODO: check if 0 or 1!
                    layer_val, layer_dep, layer_pos, memory=memory, layer_idx=1, temperature=temperature
                )

                if len(layer_val) != len(layer_dep):
                    break  # reached maximum number of tokens which can be generated

                # append sampled tokens to sequence
                val = torch.cat([val, layer_val])
                dep = torch.cat([dep, layer_dep])
                pos = torch.cat([pos, layer_pos])

        return postprocess(val, target_resolution, self.spatial_dim)
