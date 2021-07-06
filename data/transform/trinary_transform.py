from data.transform import AbstractTransform
from utils import TrinaryRepresentation


class TrinaryTransform(AbstractTransform):
    def __init__(self, spatial_dim):
        """ Creates a transform module, which transforms the input data samples for the 'discrete_transformation' embedding.

        Args:
            spatial_dim: Spatial dimensionality of input data.
        """
        super(TrinaryTransform, self).__init__()
        self.tri_repr = TrinaryRepresentation(spatial_dim)

    def transform_fx(self, value, depth, position):
        """ Transforms a single sample to trinary representation. """
        return self.tri_repr.encode_trinary(value, depth, position)
