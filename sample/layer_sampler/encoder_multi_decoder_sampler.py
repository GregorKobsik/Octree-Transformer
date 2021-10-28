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
    def __init__(self, model, embedding, head, spatial_dim, max_resolution, position_encoding, device, **_):
        """ Provides a basic implementation of the sampler for the 'encoder_only' architecture.

        Args:
            model: Model which is used for sampling.
            embedding: Token embedding type used in the model.
            head: Generative head type used in the model.
            spatial_dim: The spatial dimensionality of the array of elements.
            max_resolution: Maximum resolution the model is trained on.
            position_encoding: Defines the positional encoding of the data.
            device: Device on which, the data should be stored. Either "cpu" or "cuda" (gpu-support).
        """
        self.generators = create_token_generator(head, model, spatial_dim)
        self.compute_memory = model.compute_memory

        self.head = head
        self.spatial_dim = spatial_dim
        self.max_resolution = max_resolution
        self.pos_encoding = position_encoding
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
        val, dep, pos = preprocess(
            precondition, precondition_resolution, self.spatial_dim, self.pos_encoding, self.device
        )

        # compute the number of sampled layers and the maximum sampleable layer
        sampled_layers = len(val)
        max_layer = int(math.log2(min(target_resolution, self.max_resolution)))

        with torch.no_grad():
            memory = None

            # compute memory for already sampled layers (precondition)
            if sampled_layers >= self.num_concat_layers:
                for depth in range(self.num_concat_layers, sampled_layers + 1):
                    idx = max(0, depth - self.num_concat_layers)
                    # prepare sequence to update memory
                    if depth == self.num_concat_layers:
                        seq = self._to_sequence(torch.cat(val), torch.cat(dep), torch.cat(pos))
                    elif self.head[idx] == 'substitution':
                        seq = self._to_sequence(
                            torch.cat(val[depth - 2:depth - 1]),
                            torch.cat(dep[depth - 2:depth - 1]),
                            torch.cat(pos[depth - 2:depth - 1]),
                        )
                    elif self.head[idx] == 'double_substitution':
                        seq = self._to_sequence(
                            torch.cat(val[depth - 2:depth - 1]),
                            torch.cat(dep[depth - 2:depth - 1]),
                            torch.cat(pos[depth - 2:depth - 1]),
                        )
                    else:
                        seq = self._to_sequence(val[depth - 1], dep[depth - 1], pos[depth - 1])
                    # compute memory
                    memory = self.compute_memory(seq, memory=memory, idx=idx, is_final=False)

            # sample new tokens layer-wise
            for layer_idx in tqdm(
                range(sampled_layers + 1, max_layer + 1),
                initial=sampled_layers,
                total=max_layer,
                leave=True,
                desc="Layers",
            ):
                idx = max(0, layer_idx - self.num_concat_layers)

                kwargs = {
                    "memory": memory,
                    "idx": idx,
                    "temperature": temperature,
                }

                # init sequences for next layer
                nxt_val, nxt_dep, nxt_pos = next_layer_tokens(
                    val, dep, pos, self.spatial_dim, self.max_resolution, self.pos_encoding
                )

                # predict value tokens for current layer
                if idx == 0:
                    # sampling: encoder part
                    nxt_val = self.generators[0](val + [nxt_val], dep + [nxt_dep], pos + [nxt_pos], **kwargs)
                elif self.head[idx] == 'substitution':
                    # sampling: decoder part - 'substitution'
                    nxt_val = self.generators[idx]([val[-1], nxt_val], [dep[-1], nxt_dep], [pos[-1], nxt_pos], **kwargs)
                elif self.head[idx] == 'double_substitution':
                    # sampling: decoder part - 'substitution'
                    nxt_val = self.generators[idx](
                        [val[-2], val[-1], nxt_val], [dep[-2], dep[-1], nxt_dep], [pos[-2], pos[-1], nxt_pos], **kwargs
                    )
                else:
                    # sampling: decoder part - 'basic'
                    nxt_val = self.generators[idx]([nxt_val], [nxt_dep], [nxt_pos], **kwargs)

                # append sampled tokens to sequence
                val += [nxt_val]
                dep += [nxt_dep]
                pos += [nxt_pos]

                # prepare sequence to update memory
                if self.head[idx] == 'substitution':
                    seq = self._to_sequence(torch.cat(val[-2:]), torch.cat(dep[-2:]), torch.cat(pos[-2:]))
                elif self.head[idx] == 'double_substitution':
                    seq = self._to_sequence(torch.cat(val[-3:]), torch.cat(dep[-3:]), torch.cat(pos[-3:]))
                elif layer_idx == self.num_concat_layers:
                    seq = self._to_sequence(torch.cat(val), torch.cat(dep), torch.cat(pos))
                elif layer_idx > self.num_concat_layers:
                    seq = self._to_sequence(val[-1], dep[-1], pos[-1])
                # update memory
                if layer_idx >= self.num_concat_layers:
                    memory = self.compute_memory(seq, memory=memory, idx=idx, is_final=False)

        return postprocess(val, target_resolution, self.spatial_dim)

    def _to_sequence(self, val, dep, pos):
        """ Adds a batch dimension to the tensor and packs it into a sequence tuple. """
        return (val.unsqueeze(0), dep.unsqueeze(0), pos.unsqueeze(0))
