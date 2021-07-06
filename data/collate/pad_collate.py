from data.collate import AbstractCollate
from torch.nn.utils.rnn import pad_sequence


class PadCollate(AbstractCollate):
    def __init__(self, architecture):
        """ Creates a collate module, which pads batched sequences to equal length with the padding token '0'.

        Args:
            architecture: Defines the underlying transformer architecture.
        """
        super(PadCollate, self).__init__(architecture)

    def encoder_only(self, batch):
        """ Pads and packs a list of samples for the 'encoder_only' architecture. """
        # unpack batched sequences
        val_seq, dep_seq, pos_seq, val_tgt, dep_tgt, pos_tgt = zip(*batch)

        # pad each batched sequences with '0' to same length
        val_seq_pad = pad_sequence(val_seq, batch_first=True, padding_value=0)
        dep_seq_pad = pad_sequence(dep_seq, batch_first=True, padding_value=0)
        pos_seq_pad = pad_sequence(pos_seq, batch_first=True, padding_value=0)

        val_tgt_pad = pad_sequence(val_tgt, batch_first=True, padding_value=0)
        dep_tgt_pad = pad_sequence(dep_tgt, batch_first=True, padding_value=0)
        pos_tgt_pad = pad_sequence(pos_tgt, batch_first=True, padding_value=0)

        # return as (sequence, target)
        return (val_seq_pad, dep_seq_pad, pos_seq_pad), (val_tgt_pad, dep_tgt_pad, pos_tgt_pad)

    def encoder_decoder(self, batch):
        """ Pads and packs a list of samples for the 'encoder_decoder' architecture. """
        # unpack batched sequences
        val_enc, dep_enc, pos_enc, val_dec, dep_dec, pos_dec, val_tgt, dep_tgt, pos_tgt = zip(*batch)

        # pad each batched sequences with '0' to same length
        v_enc_pad = pad_sequence(val_enc, batch_first=True, padding_value=0)
        d_enc_pad = pad_sequence(dep_enc, batch_first=True, padding_value=0)
        p_enc_pad = pad_sequence(pos_enc, batch_first=True, padding_value=0)

        v_dec_pad = pad_sequence(val_dec, batch_first=True, padding_value=0)
        d_dec_pad = pad_sequence(dep_dec, batch_first=True, padding_value=0)
        p_dec_pad = pad_sequence(pos_dec, batch_first=True, padding_value=0)

        v_tgt_pad = pad_sequence(val_tgt, batch_first=True, padding_value=0)
        d_tgt_pad = pad_sequence(dep_tgt, batch_first=True, padding_value=0)
        p_tgt_pad = pad_sequence(pos_tgt, batch_first=True, padding_value=0)

        # return as ((input_enc, input_dec), target)
        return ((v_enc_pad, d_enc_pad, p_enc_pad), (v_dec_pad, d_dec_pad, p_dec_pad)), (v_tgt_pad, d_tgt_pad, p_tgt_pad)

    def autoencoder(self, batch):
        """ Pads and packs a list of samples for the 'autoencoder' architecture. """
        # unpack batched sequences
        val_seq, dep_seq, pos_seq, val_tgt, dep_tgt, pos_tgt = zip(*batch)

        # pad each batched sequences with '0' to same length
        val_seq_pad = pad_sequence(val_seq, batch_first=True, padding_value=0)
        dep_seq_pad = pad_sequence(dep_seq, batch_first=True, padding_value=0)
        pos_seq_pad = pad_sequence(pos_seq, batch_first=True, padding_value=0)

        val_tgt_pad = pad_sequence(val_tgt, batch_first=True, padding_value=0)
        dep_tgt_pad = pad_sequence(dep_tgt, batch_first=True, padding_value=0)
        pos_tgt_pad = pad_sequence(pos_tgt, batch_first=True, padding_value=0)

        # return as (sequence, target)
        return (val_seq_pad, dep_seq_pad, pos_seq_pad), (val_tgt_pad, dep_tgt_pad, pos_tgt_pad)
