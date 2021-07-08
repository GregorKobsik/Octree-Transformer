from utils import TrinaryRepresentation


class TrinaryTransform():
    def __init__(self, spatial_dim):
        """ Creates a transform module, which transforms the input data samples for the 'discrete_transformation' embedding.

        Args:
            spatial_dim: Spatial dimensionality of input data.
        """
        self.tri_repr = TrinaryRepresentation(spatial_dim)

    def __call__(self, value, depth, position):
        """ Transforms a single sample to trinary representation. """
        return self.tri_repr.encode_trinary(value, depth, position)
