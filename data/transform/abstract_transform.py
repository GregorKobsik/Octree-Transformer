class AbstractTransform(object):
    def __init__(self, architecture):
        """ Defines an abstract definition of the data transform function.

        Args:
            architecture: Defines which architecture is used in the transformer model.
        """
        if architecture == "encoder_only":
            if not callable(self.encoder_only):
                raise ValueError("ERROR: No `encoder_only` function implemented.")
            self.transform_fx = self.encoder_only
        elif architecture == "encoder_decoder":
            if not callable(self.encoder_decoder):
                raise ValueError("ERROR: No `encoder_decoder` function implemented.")
            self.transform_fx = self.encoder_decoder
        elif architecture == "autoencoder":
            if not callable(self.autoencoder):
                raise ValueError("ERROR: No `autoencoder` function implemented.")
            self.transform_fx = self.autoencoder
        elif architecture == "encoder_multi_decoder":
            if not callable(self.encoder_multi_decoder):
                raise ValueError("ERROR: No `encoder_multi_decoder` function implemented.")
            self.transform_fx = self.encoder_multi_decoder
        else:
            raise ValueError(f"ERROR: No transform function implemented for {architecture}")

    def __call__(self, value, depth, pos):
        """ Performs the transformation of a single sample sequence into the desired format.

        Note: Uses different output shapes for different architectures.

        Args:
            value: Raw value token sequence.
            depth: Raw depth token sequence.
            pos: Raw position token sequence.

        Return:
            Tuple representing the given sequences transformed to match the architecture requirements. The tuple has
            the shape (val, dep, pos, val, dep, pos) for the 'encoder_only' and 'autoencoder' architecture and
            (enc_value, enc_depth, enc_pos, dec_value, dec_depth, dec_pos, target) for the 'encoder_decoder'
            architecture.
        """
        return self.transform_fx(value, depth, pos)
