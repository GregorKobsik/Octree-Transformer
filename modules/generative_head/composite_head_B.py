import torch.nn as nn

from .composite_head_A import CompositeHeadA
from .linear_head import LinearHead
from .convolution_head_A import ConvolutionHeadA
from .substitution_head import SubstitutionHead
from .double_substitution_head import DoubleSubstitutionHead


class CompositeHeadB(CompositeHeadA):
    def __init__(self, num_vocab, embed_dim, resolution, spatial_dim, **_):
        """ Performs a transformation from transformer latent space into target value logits.

        Uses a different heads for each depth layer, possibly increasing the overall sequence lenght.
        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different target token values (exclusive padding token '0').
            embded_dim: Dimension of the latent embedding space of the transformer.
            resolution: Spatial resolution of sequence encoding.
            spatial_dim: Spatial dimension (2D/3D) of the sequence data.
        """
        super(CompositeHeadB, self).__init__(num_vocab, embed_dim, resolution, spatial_dim)

        kwargs = {
            "num_vocab": num_vocab,
            "embed_dim": embed_dim,
            "spatial_dim": spatial_dim,
        }

        modules = []
        if resolution >= 2:
            modules += [LinearHead(**kwargs)]
        if resolution >= 4:
            modules += [LinearHead(**kwargs)]
        if resolution >= 8:
            modules += [LinearHead(**kwargs)]
        if resolution >= 16:
            modules += [ConvolutionHeadA(**kwargs, conv_size=2**(spatial_dim - 2))]
        if resolution >= 32:
            modules += [ConvolutionHeadA(**kwargs, conv_size=2**spatial_dim)]
        if resolution >= 64:
            modules += [SubstitutionHead(**kwargs, conv_size=2**spatial_dim)]
        if resolution >= 128:
            modules += [DoubleSubstitutionHead(**kwargs, conv_size=2**spatial_dim)]

        # embeddings
        self.heads = nn.ModuleList(modules)

        self.reduction_factor = {
            1: 1,
            2: 1,
            3: 1,
            4: 2**(spatial_dim - 2),
            5: 2**spatial_dim,
            6: 2**spatial_dim,  # Note: 'substitution'
            7: 2**spatial_dim,  # Note: 'double_substitution'
        }
