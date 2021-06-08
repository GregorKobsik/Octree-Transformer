import math
import torch

from tqdm.auto import tqdm
from sample.sample_utils import next_layer_tokens

from sample.sampler import BasicEncoderDecoderSampler


class SingleConvEncoderDecoderSampler(BasicEncoderDecoderSampler):
    def __init__(self, spatial_dim, device, max_tokens, max_resolution, model):
        """ Provides a basic implementation of the sampler for the encoder decoder architecture with all basic modules.

        The following sampler works with the following combinations of modules [architecture, embedding, head]:
            - 'encoder_decoder', 'single_conv', 'single_conv'

        Args:
            spatial_dim: Spatial dimensionality of the array of elements.
            device: Device on which, the data should be stored. Either "cpu" or "cuda" (gpu-support).
            max_tokens: Maximum number of tokens a sequence can have.
            max_resolution: Maximum resolution the model is trained on
            model: Instance of model which is used for sampling.
        """
        super(SingleConvEncoderDecoderSampler, self).__init__(spatial_dim, device, max_tokens, max_resolution, model)
        self.num_sampled_tokens = 2**spatial_dim

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
        value, depth, pos = sequences

        # compute the number of finished (current) layers and the maximum sampleable layer
        cur_layer = 0 if len(depth) == 0 else int(max(depth))
        max_layer = int(math.log2(min(target_resolution, self.max_resolution)))

        with torch.no_grad():
            # sample new tokens layer by layer - encoder: finished layers, decoder: unfinished/sampled layer
            for _ in tqdm(range(cur_layer, max_layer), initial=cur_layer, total=max_layer, leave=True, desc="Layer"):

                # init sequences for next layer
                tgt_val, tgt_depth, tgt_pos = next_layer_tokens(
                    value, depth, pos, self.spatial_dim, self.max_resolution
                )

                # compute encoder output
                memory = self.process_encoder(value, depth, pos)
                # check encoder input/output length - early out if too long
                if len(memory) == self.max_tokens:
                    return value

                # sample autoregressive tokens for the next layer
                # TODO: check indizes - probably correct: range(len(tgt_val) - 1)
                for cur_token_idx in tqdm(
                    range(0, len(tgt_val), self.num_sampled_tokens), leave=False, desc="Sampling"
                ):

                    # compute decoder output
                    latent_sequence = self.process_decoder(tgt_val, tgt_depth, tgt_pos, memory, cur_token_idx)
                    # check decoder input/output length - early out if too long
                    if len(latent_sequence) == self.max_tokens:
                        return torch.cat([value, tgt_val])

                    # compute logits and sample a single token
                    logits = self.process_head(tgt_val, tgt_depth, tgt_pos, latent_sequence, cur_token_idx)
                    tgt_val = self.sample_token(tgt_val, logits, cur_token_idx, temperature)

                # append the last predicted/sampled layer to other finished layers
                value = torch.cat([value, tgt_val])
                depth = torch.cat([depth, tgt_depth])
                pos = torch.cat([pos, tgt_pos])

        return value

    def process_decoder(self, tgt_val, tgt_depth, tgt_pos, memory, token_idx):
        """ Process the decoder part of the model and return the decoded target sequence in decoder latent space.

        Args:
            tgt_val: Target value token sequence.
            tgt_depth: Target depth token sequence.
            tgt_pos: Target position token sequence.
            memory: Encoder output sequence in latent space.
            token_idx: Currently sampled token index.

        Return:
            Latent sequence vector representing target token sequence.
        """
        # decode target sequence and memory into latent space
        latent_sequence = self.model.decode(
            tgt_val[:token_idx + self.num_sampled_tokens].unsqueeze(0),  # [N, T]
            tgt_depth[:token_idx + self.num_sampled_tokens].unsqueeze(0),  # [N, T]
            tgt_pos[:token_idx + self.num_sampled_tokens].unsqueeze(0),  # [N, T, A]
            memory.unsqueeze(0),  # [N, S', E]
        )[0]  # [T', E]

        return latent_sequence

    def process_head(self, tgt_val, tgt_depth, tgt_pos, latent_sequence, token_idx):
        """ Process the head of the model to transform the latent space into logits.

        Args:
            tgt_val: Target value token sequence.
            tgt_depth: Target depth token sequence.
            tgt_pos: Target position token sequence.
            latent_sequence: Decoder output sequence in latent space.
            token_idx: Currently sampled token index.

        Return:
            Logits of already sampled tokens and currently sampled tokens.
        """
        # compute logits from latent sequence
        logits = self.model.head(
            latent_sequence[:token_idx + self.num_sampled_tokens].unsqueeze(0),  # [N, T', E]
            tgt_val[:token_idx + self.num_sampled_tokens].unsqueeze(0),  # [N, T]
            tgt_depth[:token_idx + self.num_sampled_tokens].unsqueeze(0),  # [N, T]
            tgt_pos[:token_idx + self.num_sampled_tokens].unsqueeze(0),  # [N, T, A]
        )[0]  # [T, V]

        return logits

    def sample_token(self, tgt_val, logits, token_idx, temperature):
        """ Sample autoregressiv one token from the logits and return updated target sequence.

        Args:
            tgt_val: Target value token sequence.
            logits: Logits corresponding to target sequence.
            token_idx: Currently sampled token index.
            temperature: Defines the randomness of the samples.

        Return:
            Updated target token sequence, where the sampled tokens are exchanged with a sample.
        """
        last_token_logits = logits[-self.num_sampled_tokens:]

        # compute token probabilities from logits
        probs = torch.nn.functional.softmax(last_token_logits / temperature, dim=0)  # [V]

        # zero probability for special tokens -> invalid with parent token
        probs[:, 0] = 0  # 'padding' token

        # sample next sequence token
        for i in range(self.num_sampled_tokens):
            tgt_val[token_idx + i] = torch.multinomial(probs[i], num_samples=1)[0]

        return tgt_val
