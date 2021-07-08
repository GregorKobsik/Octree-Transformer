class BasicTransform():
    """ Creates a transform module, which transforms the input data samples to pytorch tensors. """
    def __call__(self, value, depth, position):
        """ Transforms a single sample to pytorch tensors. """
        return value, depth, position
