import torch
from data.transform import BasicTransform


class PenultimateLayerTransform(BasicTransform):
    """ Creates a transform module, which transforms the input data samples for the 'substitution' embedding.

    Args:
        architecture: Defines which architecture is used in the transformer model.
    """
    def __init__(self, architecture):
        super(PenultimateLayerTransform, self).__init__(architecture)

    def encoder_decoder(self, value, depth, pos):
        """ Performs a transformation of a single sample for the 'encoder_decoder' architecture. """
        # get maximum depth layer value
        max_depth = max(depth)

        # get all sequence layers, excepth last one
        val_enc = torch.tensor(value[depth != max_depth])
        dep_enc = torch.tensor(depth[depth != max_depth])
        pos_enc = torch.tensor(pos[depth != max_depth])

        # extract penultimate and last sequence layer
        val_dec = torch.cat([torch.tensor(value[depth == (max_depth - 1)]), torch.tensor(value[depth == max_depth])])
        dep_dec = torch.cat([torch.tensor(depth[depth == (max_depth - 1)]), torch.tensor(depth[depth == max_depth])])
        pos_dec = torch.cat([torch.tensor(pos[depth == (max_depth - 1)]), torch.tensor(pos[depth == max_depth])])

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

        # extract penultimate (1) and last (2) sequence layer
        val_1 = torch.tensor(value[depth == (max_depth - 1)])
        val_2 = torch.tensor(value[depth == (max_depth)])

        dep_1 = torch.tensor(depth[depth == (max_depth - 1)])
        dep_2 = torch.tensor(depth[depth == (max_depth)])

        pos_1 = torch.tensor(pos[depth == (max_depth - 1)])
        pos_2 = torch.tensor(pos[depth == (max_depth)])

        # concat penultimate and last layer
        val_enc = torch.cat([val_1, val_2])
        dep_enc = torch.cat([dep_1, dep_2])
        pos_enc = torch.cat([pos_1, pos_2])

        # return sequences for encoder and target
        return val_enc, dep_enc, pos_enc, val_2, dep_2, pos_2
