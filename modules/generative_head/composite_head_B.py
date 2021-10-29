import torch.nn as nn

from .composite_head_A import CompositeHeadA
from .linear_head import LinearHead
from .convolution_head_A import ConvolutionHeadA
from .substitution_head import SubstitutionHead
from .double_substitution_head import DoubleSubstitutionHead


class CompositeHeadB(CompositeHeadA):
    def __init__(self, spatial_encoding, num_vocab, embed_dim, head_dim, n_layer, resolution, **_):
        """ Performs a transformation from transformer latent space into target value logits.

        Uses a different heads for each depth layer, possibly increasing the overall sequence lenght.
        Note: The token value '0' is reserved as a padding value, which does not propagate gradients.

        Args:
            num_vocab: Number of different target token values (exclusive padding token '0').
            embed_dim: Dimension of the latent embedding space of the transformer.
            head_dim: Size of embedding dimensions used in the head layers.
            n_layer: Number of layers used in each linear or convolution block.
            resolution: Spatial resolution of sequence encoding.
        """
        super(CompositeHeadB, self).__init__(spatial_encoding, num_vocab, embed_dim, head_dim, n_layer, resolution)

        kwargs = {
            "spatial_encoding": spatial_encoding,
            "num_vocab": num_vocab,
            "embed_dim": embed_dim,
            "head_dim": head_dim,
            "n_layer": n_layer
        }

        modules = []
        if resolution >= 2:
            modules += [LinearHead(**kwargs)]
        if resolution >= 4:
            modules += [LinearHead(**kwargs)]
        if resolution >= 8:
            modules += [LinearHead(**kwargs)]
        if resolution >= 16:
            modules += [ConvolutionHeadA(**kwargs, conv_size=4)]
        if resolution >= 32:
            modules += [ConvolutionHeadA(**kwargs, conv_size=8)]
        if resolution >= 64:
            modules += [SubstitutionHead(**kwargs, conv_size=8)]
        if resolution >= 128:
            modules += [DoubleSubstitutionHead(**kwargs, conv_size=8)]

        # embeddings
        self.heads = nn.ModuleList(modules)

        self.reduction_factor = {
            1: 1,
            2: 1,
            3: 1,
            4: 4,
            5: 8,
            6: 8,  # Note: 'substitution'
            7: 8,  # Note: 'double_substitution'
        }
