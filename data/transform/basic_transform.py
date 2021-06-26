import torch


class BasicTransform(object):
    def __init__(self, architecture):
        """ Creates a transform module, which transforms the input data samples for the 'basic' embedding.

        Args:
            architecture: Defines which architecture is used in the transformer model.
        """
        if architecture == "encoder_only":
            self.transform_fx = self.encoder_only
        elif architecture == "encoder_decoder":
            self.transform_fx = self.encoder_decoder
        elif architecture == "autoencoder":
            self.transform_fx = self.autoencoder
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

    def encoder_only(self, value, depth, pos):
        """ Transforms a single sample for the 'encoder_only' architecture. """
        # transform numpy arrays into pytorch tensors
        val = torch.tensor(value)
        dep = torch.tensor(depth)
        pos = torch.tensor(pos)

        # return sequences for encoder and target
        return val, dep, pos, val, dep, pos

    def encoder_decoder(self, value, depth, pos):
        """ Transforms a single sample for the 'encoder_decoder' architecture. """
        # get maximum depth layer value
        max_depth = max(depth)

        # get all sequence layers, excepth last one
        val_enc = torch.tensor(value[depth != max_depth])
        dep_enc = torch.tensor(depth[depth != max_depth])
        pos_enc = torch.tensor(pos[depth != max_depth])

        # extract last sequence layer
        val_dec = torch.tensor(value[depth == max_depth])
        dep_dec = torch.tensor(depth[depth == max_depth])
        pos_dec = torch.tensor(pos[depth == max_depth])

        # target is the last layer
        val_tgt = torch.tensor(value[depth == max_depth])
        dep_tgt = torch.tensor(depth[depth == max_depth])
        pos_tgt = torch.tensor(pos[depth == max_depth])

        # return sequences for encoder, decoder and target
        return val_enc, dep_enc, pos_enc, val_dec, dep_dec, pos_dec, val_tgt, dep_tgt, pos_tgt

    def autoencoder(self, value, depth, pos):
        """ Transforms a single sample for the 'autoencoder' architecture. """
        # get maximum depth layer value
        max_depth = max(depth)

        # transform numpy arrays into pytorch tensors
        val = torch.tensor(value[depth == max_depth])
        dep = torch.tensor(depth[depth == max_depth])
        pos = torch.tensor(pos[depth == max_depth])

        # return sequences for encoder and target
        return val, dep, pos, val, dep, pos
