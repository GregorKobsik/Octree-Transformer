import torch
import math
from tqdm.auto import tqdm

from ..token_generator import create_token_generator
from ..sample_utils import (
    next_layer_tokens,
    preprocess,
    postprocess,
)


class EncoderMultiDecoderSampler():
    def __init__(self, model, embedding, head, spatial_dim, max_resolution, device, **_):
        """ Provides a basic implementation of the sampler for the 'encoder_only' architecture.

        Args:
            model: Model which is used for sampling.
            embedding: Token embedding type used in the model.
            head: Generative head type used in the model.
            spatial_dim: The spatial dimensionality of the array of elements.
            max_resolution: Maximum resolution the model is trained on.
            device: Device on which, the data should be stored. Either "cpu" or "cuda" (gpu-support).
        """
        self.generators = create_token_generator(head, model)
        self.compute_memory = model.compute_memory

        self.spatial_dim = spatial_dim
        self.max_resolution = max_resolution
        self.num_concat_layers = 1 + int(math.log2(max_resolution)) - len(embedding)
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

        # compute the number of finished layers and the maximum sampleable layer
        cur_layer = 0 if len(dep) == 0 else int(max(dep))
        max_layer = int(math.log2(min(target_resolution, self.max_resolution)))

        with torch.no_grad():
            memory = None

            # compute memory for finished layers
            if cur_layer >= self.num_concat_layers:
                for depth in range(self.num_concat_layers, cur_layer):
                    # filter sequence
                    mask = dep <= depth if depth == self.num_concat_layers else dep == depth
                    seq = self._to_sequence(val[mask], dep[mask], pos[mask])
                    # compute memory
                    memory = self.compute_memory(seq, memory=memory, idx=0, is_final=False)

            # sample layer-wise
            for idx in tqdm(range(cur_layer, max_layer), initial=cur_layer, total=max_layer, leave=True, desc="Layers"):
                layer_idx = max(0, idx - self.num_concat_layers + 1)

                # sampling: encoder part
                if layer_idx == 0:
                    # get number of already sampld tokens
                    num_sampled = len(val)
                    # init sequences for next layer
                    lay_val, lay_dep, lay_pos = next_layer_tokens(val, dep, pos, self.spatial_dim, self.max_resolution)
                    # append future tokens to current sequence
                    val = torch.cat([val, lay_val])
                    dep = torch.cat([dep, lay_dep])
                    pos = torch.cat([pos, lay_pos])
                    # predict value tokens for current layer
                    val = self.generators[layer_idx](val, dep, pos, start_idx=num_sampled, temperature=temperature)
                    if len(val) != len(dep):
                        break  # reached maximum number of tokens which can be generated
                    # remember last 'full' sequence
                    last_val, last_dep, last_pos = val, dep, pos

                # sampling: decoder part
                else:
                    # init sequences for next layer
                    lay_val, lay_dep, lay_pos = next_layer_tokens(val, dep, pos, self.spatial_dim, self.max_resolution)
                    # predict value tokens for current layer
                    lay_val = self.generators[layer_idx](
                        lay_val, lay_dep, lay_pos, memory=memory, layer_idx=layer_idx, temperature=temperature
                    )
                    # append sampled tokens to sequence
                    val = torch.cat([val, lay_val[:len(lay_val)]])
                    dep = torch.cat([dep, lay_dep[:len(lay_val)]])
                    pos = torch.cat([pos, lay_pos[:len(lay_val)]])
                    # check maximal number of positions in transformer
                    if len(lay_val) != len(lay_dep):
                        break  # reached maximum number of tokens which can be generated
                    # remember last sequence
                    last_val, last_dep, last_pos = lay_val, lay_dep, lay_pos

                # update memory / encode last sequence
                seq = self._to_sequence(last_val, last_dep, last_pos)
                memory = self.compute_memory(seq, memory=memory, idx=layer_idx, is_final=False)

        return postprocess(val, target_resolution, self.spatial_dim)

    def _to_sequence(self, val, dep, pos):
        """ Adds a batch dimension to the tensor and packs it into a sequence tuple. """
        return (val.unsqueeze(0), dep.unsqueeze(0), pos.unsqueeze(0))
