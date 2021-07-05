from torch.nn.utils.rnn import pad_sequence


class PadCollate(object):
    def __init__(self, architecture):
        """ Creates a collate module, which pads batched sequences to equal length with the padding token '0'.

        Args:
            architecture: Defines the underlying transformer architecture.
        """
        self.architecture = architecture

    def __call__(self, batch):
        """ Pads the sequence batch and packs it into required tuples.

        Note: Uses different output shapes for different architectures architecture.

        Args:
            batch: List of transformed input sequences.

        Return:
            Tensor with padded sequences. The 'encoder_only' architecture has the shape (enc_input, target) and the
            'encoder_decoder' architecture has the shape (enc_input, dec_input, target), where 'enc/dec_input' consists
            of a (value, depth, target) tuple for the encoder or decoder, respectively.
        """
        if self.architecture == "encoder_only":
            return self.encoder_only(batch)
        elif self.architecture == "encoder_decoder":
            return self.encoder_decoder(batch)
        if self.architecture == "autoencoder":
            return self.encoder_only(batch)
        else:
            raise ValueError(f"ERROR: No padding collate function implemented for {self.architecture}")

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
