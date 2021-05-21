import math
import torch

from utils import kdTree
from tqdm.auto import tqdm
from utils import _directions


def preprocess_successive(precondition, precondition_resolution, spatial_dim, device, **_):
    """ Transform input array elements into token sequences.

    Args:
        precondition: An array of elements (pixels/voxels) as an numpy array.
        precondition_resolution: Resolution, to which the input array will be downscaled and used as a precondition for
            sampling.
        spatial_dim: The spatial dimensionality of the array of elements.
        device: The device on which, the data should be stored. Either "cpu" or "cuda" (gpu-support).

    Return:
        PyTorch tensor consisting of token sequences: (value, depth, position).
    """
    # if no input was delivered, draw a random sample from the dataset
    if input is None:
        print("ERROR: `input` cannot be `None`.")
        raise ValueError

    # convert input array into token sequence
    tree = kdTree(spatial_dim).insert_element_array(precondition)
    value, depth, pos = tree.get_token_sequence(
        depth=math.log2(precondition_resolution) + 1, return_depth=True, return_pos=True
    )  # [v/d/p, S, *]

    # convert sequence tokens to PyTorch as a long tensor
    value = torch.tensor(value, dtype=torch.long, device=device)
    depth = torch.tensor(depth, dtype=torch.long, device=device)
    pos = torch.tensor(pos, dtype=torch.long, device=device)

    return [value, depth, pos]


def sample_successive(
    sequences,
    target_resolution,
    temperature,
    model,
    spatial_dim,
    max_tokens,
    max_resolution,
    batch_first,
    **_,
):
    """ Perform an successive sampling of the given sequence until reaching the end of sequence, the maximum sequence
        length or the desired resolution.

    Args:
        sequences: Token sequences, consisting of values, depth and position sequences.
        target_resolution: Resolution up to which an object should be sampled.
        temperatur: Defines the randomness of the samples.
        model: The model which is used for sampling.
        spatial_dim: The spatial dimensionality of the sequences.
        max_tokens: The maximum number of tokens a sequence can have.
        max_resolution: The maximum resolution the model is trained on
        batch_first: Specifiy, iff the model expects the data as batch first or not.

    Return:
        A token sequence with values, encoding the final sample.
    """
    value, depth, pos = sequences
    input_len = len(value)
    remaining_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(input_len, max_tokens), initial=input_len, total=max_tokens, leave=False, desc="Sampling"):

            # append padding tokens for each new layer
            if remaining_tokens == 0:
                if 2**(max(depth) - 1) == min(max_resolution, target_resolution):
                    break  # reached desired maximum depth/resolution - early out
                value, depth, pos, remaining_tokens = append_next_layer_tokens(
                    value, depth, pos, spatial_dim, max_resolution
                )
                if remaining_tokens == 0:
                    break  # all tokens are final - early out

            # compute logits of next token
            logits = model(
                value[:i + 1].unsqueeze(1 - batch_first),  # [N, S] or [S, N]
                depth[:i + 1].unsqueeze(1 - batch_first),  # [N, S] or [S, N]
                pos[:i + 1].unsqueeze(1 - batch_first),  # [N, S, A] or [S, N, A]
            )  # [N, S, V] or [S, N, V]
            last_logit = logits[0, -1, :] if batch_first else logits[-1, 0, :]  # [V]

            # sample next sequence token
            probs = torch.nn.functional.softmax(last_logit / temperature, dim=0)  # [V]
            probs[0] = 0  # do not sample 'padding' tokens.
            value[i] = torch.multinomial(probs, num_samples=1)[0]  # TODO: check input_len == i case.

            remaining_tokens -= 1

    return value


def postprocess_successive(value, target_resolution, spatial_dim, **_):
    """ Transform sequence of value tokens into an array of elements (voxels/pixels).

        Args:
            value: Value token sequence as a pytorch tensor.
            target_resolution: Target resolution for the token sequence.
            spatial_dim: The spatial dimensionality of the value token sequence
        Return:
            An array of elements as a numpy array.
        """
    tree = kdTree(spatial_dim)
    tree = tree.insert_token_sequence(
        value.cpu().numpy(),
        resolution=target_resolution,
        autorepair_errors=True,
        silent=True,
    )
    return tree.get_element_array(mode="occupancy")


def append_next_layer_tokens(value, depth, pos, spatial_dim, max_resolution):
    """ Appends padding tokens to the value sequence, to match the neccessary shape of the next layer.
    Appends corresponding and deterministically precomputed depth and position tokens to the sequences, too.

    Args:
        value: Value token sequence given as a pytorch tensor.
        depth: Depth token sequence given as a pytorch tensor.
        pos: Position token sequence given as a pytorch tensor.
        spatial_dim: The spatial dimensionality of the value sequence.
        max_resolution: The maximal resolution the corresponding model is trained for.

    Return:
        (value, depth, pos, num_future_tokens) - Padded sequences, with the number of added tokens.
    """
    cur_device = value.device
    dirs = _directions(spatial_dim)
    num_children = 2**spatial_dim

    # got an empty input - initialize with default values and return
    if len(value) == 0:
        value = torch.tensor([1], device=cur_device, dtype=torch.long)
        depth = torch.tensor([1], device=cur_device, dtype=torch.long)
        pos = torch.ones(1, spatial_dim, device=cur_device, dtype=torch.long) * max_resolution
        num_future_tokens = torch.ones(1, device=cur_device, dtype=torch.long)
        return value, depth, pos, num_future_tokens

    # compute next layer depth and number of future tokens
    next_depth = torch.max(depth)
    num_future_tokens = num_children * torch.sum(value[depth == next_depth] == 2)

    # compute future sequence (non padding token) and future depth sequence
    nl_value = torch.tensor([1], device=cur_device, dtype=torch.long).repeat(num_future_tokens)
    nl_depth = torch.tensor([next_depth + 1], device=cur_device, dtype=torch.long).repeat(num_future_tokens)

    # retrive and copy mixed tokens positions
    pos_token = pos[torch.logical_and(value == 2, depth == next_depth)]
    nl_pos = torch.repeat_interleave(pos_token, num_children, dim=0)

    # compute position difference and add it to future positions with respect to predefined pattern
    pos_step = pos[0][0] // 2**next_depth  # assume same resolution for each dimension
    nl_pos = nl_pos + pos_step * torch.tensor(dirs, device=cur_device).repeat(pos_token.shape[0], 1)

    # concat future tokens and return
    value = torch.cat([value, nl_value])
    depth = torch.cat([depth, nl_depth])
    pos = torch.cat([pos, nl_pos])

    return value, depth, pos, num_future_tokens
