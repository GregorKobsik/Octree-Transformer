import torch
import math

from sample.sampler import AbstractSampler
from tqdm.auto import tqdm
from utils import kdTree
from sample.sample_utils import next_layer_tokens


class DoubleConvolutionalEncoderDecoderSampler(AbstractSampler):
    def __init__(self, model, embedding, head, spatial_dim, max_tokens, max_resolution, device, **_):
        """ Provides an implementation of the sampler for a convolutional encoder decoder architecture.

        The following sampler works with the following combinations of modules [architecture, embedding, head]:
            - 'encoder_decoder', 'double_conv', 'double_conv'

        Args:
            model: Model which is used for sampling.
            embedding: Token embedding type used in the model.
            head: Generative head type used in the model.
            spatial_dim: Spatial dimensionality of the array of elements.
            device: Device on which, the data should be stored. Either "cpu" or "cuda" (gpu-support).
            max_tokens: Maximum number of tokens a sequence can have.
            max_resolution: Maximum resolution the model is trained on.
        """
        super(DoubleConvolutionalEncoderDecoderSampler, self).__init__(model, embedding, head, spatial_dim, device)
        self.max_tokens = max_tokens
        self.max_resolution = max_resolution

    def preprocess(self, precondition, precondition_resolution):
        """ Transform input array elements into token sequences.

        Args:
            precondition: An array of elements (pixels/voxels) as an numpy array.
            precondition_resolution: Resolution, to which the input array will be downscaled and used as a precondition
                for sampling.

        Return:
            PyTorch tensor consisting of token sequences: (value, depth, position).
        """
        # convert input array into token sequence
        tree = kdTree(self.spatial_dim).insert_element_array(precondition)
        value, depth, pos = tree.get_token_sequence(
            depth=math.log2(precondition_resolution), return_depth=True, return_pos=True
        )

        # convert sequence tokens to PyTorch as a long tensor
        val_enc = torch.tensor(value, dtype=torch.long, device=self.device)
        dep_enc = torch.tensor(depth, dtype=torch.long, device=self.device)
        pos_enc = torch.tensor(pos, dtype=torch.long, device=self.device)

        # splitt off last layer as additional input for the decoder
        last_layer_idx = torch.argmax(dep_enc)
        val_dec = val_enc[last_layer_idx:]
        dep_dec = dep_enc[last_layer_idx:]
        pos_dec = pos_enc[last_layer_idx:]

        return (val_enc, dep_enc, pos_enc), (val_dec, dep_dec, pos_dec)

    def sample(self, sequences, target_resolution, temperature):
        """ Perform an iterative sampling of the given sequence until reaching the end of sequence, the maximum sequence
            length or the desired resolution.

        Args:
            sequences: Token sequences, consisting of values, depth and position sequences.
            target_resolution: Resolution up to which an object should be sampled.
            temperatur: Defines the randomness of the samples.

        Return:
            A token sequence with values, encoding the final sample.
        """
        seq_enc, seq_penult = sequences

        # compute the number of finished (current) layers and the maximum sampleable layer
        cur_layer = 0 if len(seq_enc[1]) == 0 else int(max(seq_enc[1]))
        max_layer = int(math.log2(min(target_resolution, self.max_resolution)))

        with torch.no_grad():
            # sample new tokens layer by layer - encoder: finished layers, decoder: unfinished/sampled layer
            # TODO: check ranges
            for _ in tqdm(range(cur_layer, max_layer), initial=cur_layer, total=max_layer, leave=True, desc="Layers"):

                # init sequences for next layer
                seq_last = next_layer_tokens(*seq_enc, self.spatial_dim, self.max_resolution)

                # compute encoder output
                memory = self.process_encoder(seq_enc)
                # check encoder input/output length - early out if too long
                if len(memory) == self.max_tokens:
                    return seq_enc[0]

                # sample autoregressive tokens for the next layer - iterate over mixed tokens of penultimate layer
                # TODO: check range - probably correct: range(sum(seq_penult[0] == 2) - 1)
                for cur_token_idx in tqdm(range(sum(seq_penult[0] == 2)), leave=False, desc="Tokens"):

                    # compute decoder output
                    latent_sequence = self.process_decoder(seq_penult, seq_last, memory, cur_token_idx)
                    # check decoder input/output length - early out if too long
                    if len(latent_sequence) == self.max_tokens:
                        return torch.cat([seq_enc[0], seq_last[0]])

                    # compute logits and sample a single token
                    logits = self.process_head(seq_penult, seq_last, latent_sequence, cur_token_idx)
                    seq_last[0] = self.sample_token(seq_last[0], logits, cur_token_idx, temperature)

                # append the last predicted/sampled layer to other finished layers
                for i in range(len(seq_enc)):
                    seq_enc[i] = torch.cat([seq_enc[i], seq_last[i]])
                seq_penult = seq_last

        return seq_enc[0]

    def process_encoder(self, seq_enc):
        """ Process the encoder part of the model and return the encoded sequence as memory in encoder latent space.

        Args:
            seq_enc:

        Return:
            TODO: extend description
        """
        value, depth, pos = seq_enc

        # precompute encoder memory / process input values sequence
        memory = self.model.encode(
            value.unsqueeze(0),  # [N, S]
            depth.unsqueeze(0),  # [N, S]
            pos.unsqueeze(0),  # [N, S, A]
        )[0]  # [N, S', E]

        return memory

    def process_decoder(self, seq_penult, seq_last, memory, token_idx):
        """ Process the decoder part of the model and return the decoded target sequence in decoder latent space.

        Args:
            seq_penult:
            seq_last:
            memory:
            token_idx:

        Return:
            TODO: extend description
        """
        # add one more token, as the decoder input sequence is shifted by the sos token
        token_idx += 1

        # get indices of penultimate layer with a mixed token
        idx = torch.nonzero(seq_penult[0] == 2)

        # compute the corresponding subsequence for the penultimate and last layer
        idx_penult = idx[token_idx]
        idx_last = token_idx * 2**self.spatial_dim

        # concat the penultimate and last sequences as input for the decoder
        tgt_val = torch.cat([seq_penult[0][:idx_penult], seq_last[0][:idx_last]])
        tgt_dep = torch.cat([seq_penult[1][:idx_penult], seq_last[1][:idx_last]])
        tgt_pos = torch.cat([seq_penult[2][:idx_penult], seq_last[2][:idx_last]])

        # decode target sequence and memory into latent space
        latent_sequence = self.model.decode(
            tgt_val.unsqueeze(0),  # [N, T]
            tgt_dep.unsqueeze(0),  # [N, T]
            tgt_pos.unsqueeze(0),  # [N, T, A]
            memory.unsqueeze(0),  # [N, S', E]
        )[0]  # [T', E]

        return latent_sequence

    def process_head(self, seq_penult, seq_last, latent_sequence, token_idx):
        """ Process the head of the model to transform the latent space into logits.

        Args:
            seq_penult:
            seq_last:
            latent_sequence:
            token_idx:

        Return:
            TODO: extend description
        """
        # add one more token, as the decoder input sequence is shifted by the sos token
        token_idx += 1

        # get indices of penultimate layer with a mixed token
        idx = torch.nonzero(seq_penult[0] == 2)

        # compute the corresponding subsequence for the penultimate layer
        idx_penult = idx[token_idx]

        # concat the penultimate and last sequences as input for the head (last layer is discarded in head)
        tgt_val = torch.cat([seq_penult[0][:idx_penult], seq_last[0]])
        tgt_dep = torch.cat([seq_penult[1][:idx_penult], seq_last[1]])
        tgt_pos = torch.cat([seq_penult[2][:idx_penult], seq_last[2]])

        # compute logits from latent sequence
        logits = self.model.head(
            latent_sequence.unsqueeze(0),  # [N, T', E]
            tgt_val.unsqueeze(0),  # [N, T]
            tgt_dep.unsqueeze(0),  # [N, T]
            tgt_pos.unsqueeze(0),  # [N, T, A]
        )[0]  # [T, V]

        return logits

    def sample_token(self, tgt_val, logits, token_idx, temperature):
        """ Sample autoregressiv one token from the logits and return updated target sequence.

        Args:
            tgt_val:
            logits:
            token_idx:
            temperature:

        Return:
            TODO: extend description
        """
        # compute number of target sequence tokens, that will be sampled at once
        num_tokens = 2**self.spatial_dim

        # extract corresponding logits
        token_logits = logits[-num_tokens:]  # [2**A, V]

        # compute token probabilities from logits
        probs = torch.nn.functional.softmax(token_logits / temperature, dim=1)  # [2**A, V]

        # zero probability for special tokens -> invalid with parent token
        probs[:, 0] = 0  # 'padding' token

        # sample next sequence token
        for i in range(num_tokens):
            tgt_val[token_idx + i] = torch.multinomial(probs[i], num_samples=1)[0]

        return tgt_val
