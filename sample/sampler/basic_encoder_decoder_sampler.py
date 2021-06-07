import math
import torch

from tqdm.auto import tqdm
from utils import kdTree
from sample.sample_utils import next_layer_tokens


class BasicEncoderDecoderSampler():
    def __init__(self, spatial_dim, device, max_tokens, max_resolution, model):
        """ Provides a basic implementation of the sampler for the encoder decoder architecture with all basic modules.

        The following sampler works with the following combinations of modules [architecture, embedding, head]:
            - 'encoder_decoder', 'basic', 'generative_basic'

        Args:
            spatial_dim: Spatial dimensionality of the array of elements.
            device: Device on which, the data should be stored. Either "cpu" or "cuda" (gpu-support).
            max_tokens: Maximum number of tokens a sequence can have.
            max_resolution: Maximum resolution the model is trained on
            model: Instance of model which is used for sampling.
        """
        self.spatial_dim = spatial_dim
        self.device = device
        self.max_tokens = max_tokens
        self.max_resolution = max_resolution
        self.model = model

    def __call__(self, precondition, precondition_resolution, target_resolution, temperature):
        """ Sample a single example based on the given `precondition` up to `target_resolution`.

        Args:
            precondition: An array of elements (pixels/voxels) as an numpy array.
            precondition_resolution: Resolution, to which the input array will be downscaled and used as a precondition
                for sampling.
            target_resolution: Resolution up to which an object should be sampled.
            temperature: Defines the randomness of the samples.

        Return:
            An array of elements as a numpy array, which represents the sampled shape.
        """
        # preprocess the input and transform it into a token sequence
        sequence = self.preprocess(precondition, precondition_resolution)

        # enhance the resolution of the sequence or generate a new sequence by sampling new token values
        value = self.sample(sequence, target_resolution, temperature)

        # postprocess the token value sequence and return it as an array of elements
        return self.postprocess(value, target_resolution)

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
        value = torch.tensor(value, dtype=torch.long, device=self.device)
        depth = torch.tensor(depth, dtype=torch.long, device=self.device)
        pos = torch.tensor(pos, dtype=torch.long, device=self.device)

        return value, depth, pos

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
                for cur_token_idx in tqdm(range(len(tgt_val)), leave=False, desc="Sampling"):

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

    def process_encoder(self, value, depth, pos):
        """ Process the encoder part of the model and return the encoded sequence as memory in encoder latent space.

        Args:
            value:
            depth:
            pos:

        Return:
            TODO: extend description
        """
        # precompute encoder memory / process input values sequence
        memory = self.model.encode(
            value.unsqueeze(0),  # [N, S]
            depth.unsqueeze(0),  # [N, S]
            pos.unsqueeze(0),  # [N, S, A]
        )[0]  # [N, S', E]

        return memory

    def process_decoder(self, tgt_val, tgt_depth, tgt_pos, memory, token_idx):
        """ Process the decoder part of the model and return the decoded target sequence in decoder latent space.

        Args:
            tgt_val:
            tgt_depth:
            tgt_pos:
            memory:
            token_idx:

        Return:
            TODO: extend description
        """
        # decode target sequence and memory into latent space
        latent_sequence = self.model.decode(
            tgt_val[:token_idx + 1].unsqueeze(0),  # [N, T]
            tgt_depth[:token_idx + 1].unsqueeze(0),  # [N, T]
            tgt_pos[:token_idx + 1].unsqueeze(0),  # [N, T, A]
            memory.unsqueeze(0),  # [N, S', E]
        )[0]  # [T', E]

        return latent_sequence

    def process_head(self, tgt_val, tgt_depth, tgt_pos, latent_sequence, token_idx):
        """ Process the head of the model to transform the latent space into logits.

        Args:
            tgt_val:
            tgt_depth:
            tgt_pos:
            latent_sequence:
            token_idx:

        Return:
            TODO: extend description
        """
        # compute logits from latent sequence
        logits = self.model.head(
            latent_sequence[:token_idx + 1].unsqueeze(0),  # [N, T', E]
            tgt_val[:token_idx + 1].unsqueeze(0),  # [N, T]
            tgt_depth[:token_idx + 1].unsqueeze(0),  # [N, T]
            tgt_pos[:token_idx + 1].unsqueeze(0),  # [N, T, A]
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
        last_token_logits = logits[-1]

        # compute token probabilities from logits
        probs = torch.nn.functional.softmax(last_token_logits / temperature, dim=0)  # [V]

        # zero probability for special tokens -> invalid with parent token
        probs[0] = 0  # 'padding' token

        # sample next sequence token
        tgt_val[token_idx] = torch.multinomial(probs, num_samples=1)[0]
        return tgt_val

    def postprocess(self, value, target_resolution):
        """ Transform sequence of value tokens into an array of elements (voxels/pixels).

        Args:
            value: Value token sequence as a pytorch tensor.
            target_resolution: Resolution up to which an object should be sampled.

        Return:
            An array of elements as a numpy array.
        """
        tree = kdTree(self.spatial_dim)
        tree = tree.insert_token_sequence(
            value.cpu().numpy(),
            resolution=target_resolution,
            autorepair_errors=True,
            silent=True,
        )
        return tree.get_element_array(mode="occupancy")
