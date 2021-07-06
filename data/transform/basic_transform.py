from data.transform import AbstractTransform


class BasicTransform(AbstractTransform):
    def __init__(self):
        """ Creates a transform module, which transforms the input data samples to pytorch tensors. """
        super(BasicTransform, self).__init__()

    def transform_fx(self, value, depth, position):
        """ Transforms a single sample to pytorch tensors. """
        return value, depth, position
