import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .linear_head import LinearHead
from .half_conv_head_A import HalfConvolutionalHeadA
from .single_conv_head_A import SingleConvolutionalHeadA
from .substitution_head import SubstitutionHead


class CompositeHeadA(nn.Module):
    def __init__(self, num_vocab, embed_dim, spatial_dim):
        """ Performs a transformation from transformer latent space into target value logits.

        Uses a different heads for each depth layer, possibly increasing the overall sequence lenght.
        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different target token values (exclusive padding token '0').
            embded_dim: Dimension of the latent embedding space of the transformer.
            spatial_dim: Spatial dimension (2D/3D) of the sequence data.
        """
        super(CompositeHeadA, self).__init__()

        kwargs = {
            "num_vocab": num_vocab,
            "embed_dim": embed_dim,
            "spatial_dim": spatial_dim,
        }

        # embeddings
        self.heads = nn.ModuleList(
            [
                LinearHead(**kwargs),
                LinearHead(**kwargs),
                LinearHead(**kwargs),
                HalfConvolutionalHeadA(**kwargs),
                SingleConvolutionalHeadA(**kwargs),
                SubstitutionHead(**kwargs),
            ]
        )

        self.reduction_factor = {
            1: 1,
            2: 1,
            3: 1,
            4: 2**(spatial_dim - 1),
            5: 2**spatial_dim,
            6: 2**spatial_dim,  # Note: 'substitution'
        }

    def forward(self, x, value, depth, position):
        """ Transforms the output of the transformer target value logits.

        Args:
            x: Output of the transformer, the latent vector [N, T, E].
            value: Target value token sequence [N, T].
            depth: Target depth token sequence [N, T].
            position: Target position token sequence [N, T, A].

        Return
            Logits of target value sequence.
        """
        batch_depth = torch.max(depth)
        out = []

        # process each sample individually
        for latent_vec, val, dep, pos in zip(x, value, depth, position):
            logits = torch.tensor([], device=x.device)
            vector_idx = 0

            # compute logits layerwise
            for layer_idx, head in enumerate(self.heads):
                layer_depth = layer_idx + 1
                if layer_depth > batch_depth:
                    break  # reached max depth layer

                if layer_depth < 6:
                    # get value, depth and position sequence of current layer
                    layer_val = val[dep == layer_depth]
                    layer_dep = dep[dep == layer_depth]
                    layer_pos = pos[dep == layer_depth]
                    # compute number of vectors in latent vector of current layer
                    num_vectors = torch.sum(dep == layer_depth) // self.reduction_factor[layer_depth]
                else:  # handle substitution
                    # get value, depth and position sequence of previous and current layer
                    layer_val = torch.cat([val[dep == (layer_depth - 1)], val[dep == layer_depth]])
                    layer_dep = torch.cat([dep[dep == (layer_depth - 1)], dep[dep == layer_depth]])
                    layer_pos = torch.cat([pos[dep == (layer_depth - 1)], pos[dep == layer_depth]])
                    # compute number of vectors in latent vector of current layer
                    num_vectors = torch.sum(dep == (layer_depth - 1)) // self.reduction_factor[layer_depth]

                # filter latent vector of current layer
                layer_vec = latent_vec[vector_idx:vector_idx + num_vectors]

                # compute layer logits
                layer_logits = head(
                    layer_vec.unsqueeze(0),
                    layer_val.unsqueeze(0),
                    layer_dep.unsqueeze(0),
                    layer_pos.unsqueeze(0),
                )[0]
                logits = torch.cat([logits, layer_logits])

                # discard processed tokens
                vector_idx += num_vectors

            out += [logits]

        # pad embedding sequence
        return pad_sequence(out, batch_first=True, padding_value=0.0)
