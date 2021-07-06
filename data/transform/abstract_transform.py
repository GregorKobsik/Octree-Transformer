class AbstractTransform(object):
    def __init__(self):
        """ Defines an abstract definition of the data transform function. """

    def __call__(self, value, depth, position):
        """ Performs the transformation of a single sample sequence into the desired format.

        Note: Uses different output shapes for different architectures.

        Args:
            value: Raw value token sequence.
            depth: Raw depth token sequence.
            position: Raw position token sequence.

        Return:
            Tuple with transformed (value, depth, position).
        """
        return self.transform_fx(value, depth, position)
