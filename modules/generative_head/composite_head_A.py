import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .linear_head import LinearHead
from .half_conv_head_A import HalfConvolutionalHeadA
from .single_conv_head_A import SingleConvolutionalHeadA
# from .substitution_head import SubstitutionHead


class CompositeHeadA(nn.Module):
    def __init__(self, num_vocab, embed_dim, spatial_dim):
        """ Performs a transformation from transformer latent space into target value logits.

        Uses a different heads for each depth layer, possibly increasing the overall sequence lenght.
        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different target token values (exclusive padding token '0').
            embded_dim: Dimension of the latent embedding space of the transformer.
            spatial_dim: Spatial dimension (2D/3D) of the sequence data.

        TODO: implement 'substitution'
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
                # SubstitutionHead(**kwargs),
            ]
        )

        self.reduction_factor = {
            1: 1,
            2: 1,
            3: 1,
            4: 2**(spatial_dim - 1),
            5: 2**spatial_dim,
            6: 2**spatial_dim**2,
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
        batch_size = len(value)

        out = []

        # process each sample individually
        for i in range(batch_size):
            # extract depth sequence of current sample
            sample_x, dep = x[i], depth[i]
            logits = torch.tensor([], device=x.device)

            # compute logits layerwise
            for layer_idx, head in enumerate(self.heads):
                layer_depth = layer_idx + 1
                if layer_depth > batch_depth:
                    break  # reached max depth layer

                # compute number of tokens
                num_tokens = torch.sum(dep == layer_depth) // self.reduction_factor[layer_depth]

                # TODO: handle layers for substitution!
                # compute layer logits
                layer_logits = head(sample_x[:num_tokens].unsqueeze(0), None, None, None)[0]
                logits = torch.cat([logits, layer_logits])

                # discard processed tokens
                sample_x = sample_x[num_tokens:]

            out += [logits]

        # pad embedding sequence
        return pad_sequence(out, batch_first=True, padding_value=0.0)
