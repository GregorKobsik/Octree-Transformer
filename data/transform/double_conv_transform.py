import torch
from data.transform import BasicTransform


class DoubleConvolutionalTransform(BasicTransform):
    """ Creates a transform module, which transforms the input data samples for the 'double_conv' embedding.

    Args:
        architecture: Defines whether the transformer uses a 'encoder_only' or 'encoder_decocer' architecture.
    """
    def __init__(self, architecture):
        super(DoubleConvolutionalTransform, self).__init__(architecture)

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
