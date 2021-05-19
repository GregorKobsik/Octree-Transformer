import torch
import math
from tqdm.auto import tqdm
from modules import ShapeTransformer
from utils.kd_tree import kdTree, _directions


class Sampler():
    def __init__(self, checkpoint_path: str, device="cuda"):
        """ Initializes the sampler class. Loads the correct model and sets functions and parameters according to the
            given model.

        Args:
            checkpoint_path: Relative or absolute path to a checkpoint file ("*.ckpt") containing a trained model.
            device: Selects the device on which the sampling should be performed. Either "cpu" or "cuda" (gpu-support)
                available.
        """
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.device = device

        # load and restore model from checkpoint
        pl_module = ShapeTransformer.load_from_checkpoint(checkpoint_path)
        pl_module.freeze()
        self.model = pl_module.model.eval().to(device)

        # extract hyperparameters from the model
        hparams = pl_module.hparams

        self.max_resolution = 2**(hparams["tree_depth"] - 1)
        self.max_tokens = hparams["num_positions"]
        self.dataset = hparams["dataset"]
        self.subclass = hparams["subclass"]

        self.batch_first = False if hparams['attention'].startswith("basic") else True
        self.iterative = True if hparams['architecture'] == "encoder_decoder" else False
        self.spatial_dim = hparams['spatial_dim']
        self.dirs = _directions(self.spatial_dim)

        # prepare sampling functions
        self.preprocess_function = self.preprocess
        self.sample_function = self.sample_iterative if self.iterative else self.sample_successive
        self.postprocess_function = self.postprocess_iterative if self.iterative else self.postprocess_successive

    def sample_preconditioned(self, precondition, precondition_resolution=1, target_resolution=32, temperature=1.0):
        """ Samples a single array of elements from the model.

        Args:
            precondition: Use a precondition for sampling up to a resolution in 'start_resolution' given as a numpy
                element array.
            precondition_resolution: Initial resolution used for the precondition. The precondition should have at
                least the `precondition_resolution` side length. Resolution values can be only power of 2.
            target_resolution: The target resolution for the up-sampling process. The resolution should be not bigger,
                than the maximal trained model resolution. Resolution values can be only power of 2.
            temperature: Defines the randomness of the samples. Lower temperatures make the model increasingly
                confident in its top choices, while temperatures greater than 1 decrease confidence. 0 temperature is
                equivalent to argmax/max likelihood, while infinite temperature corresponds to a uniform sampling.

        Return:
            A sampled array of elements (pixels/voxels) with the size of `target_resolution` as a numpy array.

        TODO: If `input` is None, than a single random sample will be drawn from the dataset used in the model.
        TODO: extend with `batch_size` to allow parallel sampling
        """
        # preprocess the input and transform it into a token sequence
        sequence = self.preprocess_function(precondition, precondition_resolution)

        # enhance the resolution of the sequence or generate a new sequence by sampling new token values
        value = self.sample_function(sequence, target_resolution, temperature)

        # postprocess the token value sequence and return it as an array of elements
        return self.postprocess_function(value, target_resolution)

    def sample_random(self, target_resolution=32, temperature=1.0):
        """ Sample a single unconditioned random array of elements from the model.

        Args:
            target_resolution: The target resolution for the up-sampling process. The resolution should be not bigger,
                than the maximal trained model resolution. Resolution values can be only power of 2.
            temperature: Defines the randomness of the samples. Lower temperatures make the model increasingly
                confident in its top choices, while temperatures greater than 1 decrease confidence. 0 temperature is
                equivalent to argmax/max likelihood, while infinite temperature corresponds to a uniform sampling.

        Return:
            A sampled array of elements (pixels/voxels) with the size of `target_resolution` as a numpy array.
        """
        # create an initial array, with all elements marked as undefined/mixed.
        initial_element_array = torch.full(self.spatial_dim * [1], 2, dtype=torch.long).numpy()
        return self.sample(initial_element_array, 1, target_resolution, temperature)

    def preprocess(self, input, start_resolution):
        """ Transform input array elements into token sequences.

        Args:
            input: An array of elements (pixels/voxels) as an numpy array.
            start_resolution: Resolution, to which the input array will be downscaled and used as a precondition for
                sampling.

        Return:
            PyTorch tensor consisting of token sequences: (value, depth, position).
        """
        # if no input was delivered, draw a random sample from the dataset
        if input is None:
            print("ERROR: `input` cannot be `None`.")
            raise ValueError

        # convert input array into token sequence
        tree = kdTree(self.spatial_dim).insert_element_array(input)
        value, depth, pos = tree.get_token_sequence(
            depth=math.log2(start_resolution) + 1, return_depth=True, return_pos=True
        )  # [v/d/p, S, *]

        # convert sequence tokens to PyTorch as a long tensor
        value = torch.tensor(value, dtype=torch.long, device=self.device)
        depth = torch.tensor(depth, dtype=torch.long, device=self.device)
        pos = torch.tensor(pos, dtype=torch.long, device=self.device)

        return [value, depth, pos]

    def sample_successive(self, sequences, target_resolution, temperature):
        """ Perform a successive sampling of the given sequence until reaching the end of sequence of maximum sequence
            length.

        Args:
            sequences: Token sequences, consisting of values, depth and position sequences.
            target_resolution: Resolution up to which an object should be sampled.
            temperatur: Defines the randomness of the samples.

        Return:
            A token sequence with values, encoding the final sample.
        """
        value, depth, pos = sequences
        input_len = len(value)
        remaining_tokens = 0

        with torch.no_grad():
            for i in tqdm(range(input_len, self.max_tokens), leave=False, desc="Sampling"):

                # append padding tokens for each new layer
                if remaining_tokens == 0:
                    if 2**(max(depth) - 1) == min(self.max_resolution, target_resolution):
                        break  # reached desired maximum depth/resolution - early out
                    value, depth, pos, remaining_tokens = self.append_next_layer_tokens(value, depth, pos)
                    if remaining_tokens == 0:
                        break  # all tokens are final - early out

                # compute logits of next token
                logits = self.model(
                    value[:i + 1].unsqueeze(1 - self.batch_first),  # [N, S] or [S, N]
                    depth[:i + 1].unsqueeze(1 - self.batch_first),  # [N, S] or [S, N]
                    pos[:i + 1].unsqueeze(1 - self.batch_first),  # [N, S, A] or [S, N, A]
                )  # [N, S, V] or [S, N, V]
                last_logit = logits[0, -1, :] if self.batch_first else logits[-1, 0, :]  # [V]

                # sample next sequence token
                probs = torch.nn.functional.softmax(last_logit / temperature, dim=0)  # [V]
                probs[0] = 0  # do not sample 'padding' tokens.
                value[i] = torch.multinomial(probs, num_samples=1)[0]  # TODO: check input_len == i case.

                remaining_tokens -= 1

        return value

    def sample_iterative(self, sequences, temperature):
        """ TODO """
        return

    def postprocess_successive(self, value, target_resolution):
        """ Transform sequence of value tokens into an array of elements (voxels/pixels).

        Args:
            value: Value token sequence as a pytorch tensor.
            target_resolution: Target resolution for the token sequence.

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

    def postprocess_iterative(self, value_sequence):
        """ TODO """
        return

    def append_next_layer_tokens(self, value, depth, pos):
        """ Appends padding tokens to the value sequence, to match the neccessary shape of the next layer.
        Appends corresponding and deterministically precomputed depth and position tokens to the sequences, too.

        Args:
            value: Value token sequence given as a pytorch tensor.
            depth: Depth token sequence given as a pytorch tensor.
            pos: Position token sequence given as a pytorch tensor.

        Return:
            (value, depth, pos, num_future_tokens) - Padded sequences, with the number of added tokens.
        """
        # got an empty input - initialize with default values and return
        if len(value) == 0:
            value = torch.tensor([0], device=self.device, dtype=torch.long)
            depth = torch.tensor([1], device=self.device, dtype=torch.long)
            pos = torch.ones(self.spatial_dim, 1, device=self.device, dtype=torch.long) * self.max_resolution
            num_future_tokens = torch.ones(1, device=self.device, dtype=torch.long)
            return value, depth, pos, num_future_tokens

        # compute next layer depth and number of future tokens
        next_depth = torch.max(depth)
        num_future_tokens = 2**self.spatial_dim * torch.sum(value[depth == next_depth] == 2)

        # compute future sequence (as padding) and future depth sequence
        nl_value = torch.tensor([0], device=self.device, dtype=torch.long).repeat(num_future_tokens)
        nl_depth = torch.tensor([next_depth + 1], device=self.device, dtype=torch.long).repeat(num_future_tokens)

        # retrive and copy mixed tokens positions
        pos_token = pos[torch.logical_and(value == 2, depth == next_depth)]
        nl_pos = torch.repeat_interleave(pos_token, 2**self.spatial_dim, dim=0)

        # compute position difference and add it to future positions with respect to predefined pattern
        pos_step = pos[0][0] // 2**next_depth  # assume same resolution for each dimension
        nl_pos = nl_pos + pos_step * torch.tensor(self.dirs, device=self.device).repeat(pos_token.shape[0], 1)

        # concat future tokens and return
        value = torch.cat([value, nl_value])
        depth = torch.cat([depth, nl_depth])
        pos = torch.cat([pos, nl_pos])

        return value, depth, pos, num_future_tokens
