import torch
from utils import TrinaryRepresentation


class TrinaryTransform(object):
    def __init__(self, architecture, spatial_dim):
        """ Creates a transform module, which transforms the input data samples for the 'basic' embedding.

        Args:
            architecture: Defines which architecture is used in the transformer model.
            spatial_dim: Spatial dimensionality of input data.
        """
        self.architecture = architecture
        self.tri_repr = TrinaryRepresentation(spatial_dim)

    def __call__(self, value, depth, pos):
        """ Performs the transformation of a single sample sequence into the desired format.

        Note: Uses different output shapes for 'encoder_only' and 'encoder_decoder' architecture.

        Args:
            value: Raw value token sequence.
            depth: Raw depth token sequence.
            pos: Raw position token sequence.

        Return:
            Tuple representing the given sequences transformed to match the architecture requirements. The tuple has
            the shape (value, depth, pos, target) for the 'encoder_only' architecture and (enc_value, enc_depth,
            enc_pos, dec_value, dec_depth, dec_pos, target) for the 'encoder_decoder' architecture.
        """
        if self.architecture == "encoder_only":
            return self.encoder_only(value, depth, pos)
        elif self.architecture == "encoder_decoder":
            return self.encoder_decoder(value, depth, pos)
        elif self.architecture == "autoencoder":
            return self.autoencoder(value, depth, pos)
        else:
            raise ValueError(f"ERROR: No transform function implemented for {self.architecture}")

    def encoder_only(self, value, depth, pos):
        """ Transforms a single sample for the 'encoder_only' architecture. """
        # encoder sequences into trinary representation
        val_tri, dep_tri, pos_tri = self.tri_repr.encode_trinary(value, depth, pos)

        # transform numpy arrays into pytorch tensors
        val = torch.tensor(val_tri)
        dep = torch.tensor(dep_tri)
        pos = torch.tensor(pos_tri)

        # return sequences for encoder and target
        return val, dep, pos, val, dep, pos

    def encoder_decoder(self, value, depth, pos):
        """ Transforms a single sample for the 'encoder_decoder' architecture. """
        # encoder sequences into trinary representation
        val_tri, dep_tri, pos_tri = self.tri_repr.encode_trinary(value, depth, pos)

        # get maximum depth layer value
        max_depth = max(dep_tri)

        # get all sequence layers, excepth last one
        val_enc = torch.tensor(val_tri[dep_tri != max_depth])
        dep_enc = torch.tensor(dep_tri[dep_tri != max_depth])
        pos_enc = torch.tensor(pos_tri[dep_tri != max_depth])

        # extract last sequence layer
        val_dec = torch.tensor(val_tri[dep_tri == max_depth])
        dep_dec = torch.tensor(dep_tri[dep_tri == max_depth])
        pos_dec = torch.tensor(pos_tri[dep_tri == max_depth])

        # target is the last layer
        val_tgt = torch.tensor(val_tri[dep_tri == max_depth])
        dep_tgt = torch.tensor(dep_tri[dep_tri == max_depth])
        pos_tgt = torch.tensor(pos_tri[dep_tri == max_depth])

        # return sequences for encoder, decoder and target
        return val_enc, dep_enc, pos_enc, val_dec, dep_dec, pos_dec, val_tgt, dep_tgt, pos_tgt

    def autoencoder(self, value, depth, pos):
        """ Transforms a single sample for the 'autoencoder' architecture. """
        # encoder sequences into trinary representation
        val_tri, dep_tri, pos_tri = self.tri_repr.encode_trinary(value, depth, pos)

        # get maximum depth layer value
        max_depth = max(dep_tri)

        # transform numpy arrays into pytorch tensors
        val = torch.tensor(val_tri[dep_tri == max_depth])
        dep = torch.tensor(dep_tri[dep_tri == max_depth])
        pos = torch.tensor(pos_tri[dep_tri == max_depth])

        # return sequences for encoder and target
        return val, dep, pos, val, dep, pos
