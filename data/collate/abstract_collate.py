class AbstractCollate(object):
    def __init__(self, architecture):
        """ Defines an abstract definition of the data collate function.

        Args:
            architecture: Defines which architecture is used in the transformer model.
        """
        if architecture == "encoder_only":
            if not callable(self.encoder_only):
                raise ValueError("ERROR: No `encoder_only` function implemented.")
            self.collate_fx = self.encoder_only
        elif architecture == "encoder_decoder":
            if not callable(self.encoder_decoder):
                raise ValueError("ERROR: No `encoder_decoder` function implemented.")
            self.collate_fx = self.encoder_decoder
        elif architecture == "autoencoder":
            if not callable(self.autoencoder):
                raise ValueError("ERROR: No `autoencoder` function implemented.")
            self.collate_fx = self.autoencoder
        elif architecture == "encoder_multi_decoder":
            if not callable(self.encoder_multi_decoder):
                raise ValueError("ERROR: No `encoder_multi_decoder` function implemented.")
            self.collate_fx = self.encoder_multi_decoder
        else:
            raise ValueError(f"ERROR: No collate function implemented for {architecture}")

    def __call__(self, batch):
        """ Transforms the sequence batch and packs it into required tuples.

        Note: Uses different output shapes for different architectures architecture.

        Args:
            batch: List of transformed input sequences.

        Return:
            Tensor with layerwise packed sequences.
        """
        return self.collate_fx(batch)
