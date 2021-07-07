import torch
from torch.nn.utils.rnn import pad_sequence


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

    def _to_sequences(self, batch):
        """ Transform a list on numpy arrays into sequences of pytorch tensors. """
        batch = [(torch.tensor(v), torch.tensor(d), torch.tensor(p)) for v, d, p in batch]

        # unpack batched sequences
        return zip(*batch)

    def _pad_batch(self, batch):
        """ Unpack batch and pad each sequence to a tensor of equal length. """
        val, dep, pos = self._to_sequences(batch)

        # pad each sequence
        val_pad = pad_sequence(val, batch_first=True, padding_value=0)
        dep_pad = pad_sequence(dep, batch_first=True, padding_value=0)
        pos_pad = pad_sequence(pos, batch_first=True, padding_value=0)

        return val_pad, dep_pad, pos_pad
