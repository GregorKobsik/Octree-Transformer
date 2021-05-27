import math
import torch

from utils import kdTree, RepresentationTransformator
from tqdm.auto import tqdm


def preprocess_iterative(precondition, precondition_resolution, spatial_dim, device, **_):
    """ Transform input array elements into token sequences.

    Args:
        precondition: An array of elements (pixels/voxels) as an numpy array.
        precondition_resolution: Resolution, to which the input array will be downscaled and used as a precondition for
            sampling.
        spatial_dim: The spatial dimensionality of the array of elements.
        device: The device on which, the data should be stored. Either "cpu" or "cuda" (gpu-support).

    Return:
        PyTorch tensor consisting of token sequences: (value, depth, position, target).
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
    target = torch.tensor([], dtype=torch.long, device=device)

    return [value, depth, pos, target]


def sample_iterative(
    sequences,
    target_resolution,
    temperature,
    model,
    spatial_dim,
    max_tokens,
    max_resolution,
    batch_first,
    device,
    **_,
):
    """ Perform an iterative sampling of the given sequence until reaching the end of sequence, the maximum sequence
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

    value, depth, pos, target = sequences
    input_len = len(value)
    remaining_tokens = 0
    rep_trans = RepresentationTransformator(spatial_dim)

    with torch.no_grad():
        for i in tqdm(range(input_len, max_tokens), initial=input_len, total=max_tokens, leave=False, desc="Sampling"):

            # decode the target sequence after each depth layer
            if remaining_tokens == 0:
                if len(target) != 0:
                    value, depth, pos = rep_trans.iterative_to_successive_pytorch(value, depth, pos, target)
                if 2**(max(depth) - 1) == min(max_resolution, target_resolution):
                    break  # reached desired maximum depth/resolution - early out
                if len(depth[torch.logical_and(depth == max(depth), value == 2)]) == 0:
                    break  # all tokens are final - early out

                # reset target sequence
                remaining_tokens = len(depth[depth == max(depth)])
                target = torch.tensor(remaining_tokens * [1], dtype=torch.long, device=target.device)
                cur_token = 0

                # precompute encoder memory / process input values sequence
                memory = model.encode(
                    value.unsqueeze(1 - batch_first),  # [N, S] or [S, N]
                    depth.unsqueeze(1 - batch_first),  # [N, S] or [S, N]
                    pos.unsqueeze(1 - batch_first),  # [N, S, A] or [S, N, A]
                )  # [N, S, V] or [S, N, V]

                # prefetch depth and position sequences for the decoder
                tgt_idx = torch.argmax(depth)
                tgt_depth = depth[tgt_idx:] + 1  # [T]
                tgt_pos = pos[tgt_idx:]  # [T, A]

            # compute logits of next token
            logits = model.decode(
                target[:cur_token + 1].unsqueeze(1 - batch_first),  # [N, S] or [S, N]
                tgt_depth[:cur_token + 1].unsqueeze(1 - batch_first),  # [N, S] or [S, N]
                tgt_pos[:cur_token + 1].unsqueeze(1 - batch_first),  # [N, S, A] or [S, N, A]
                memory
            )  # [N, S, V] or [S, N, V]
            last_logit = logits[0, -1, :] if batch_first else logits[-1, 0, :]  # [V]

            # compute token probabilities from logits
            probs = torch.nn.functional.softmax(last_logit / temperature, dim=0)  # [V]

            # zero probability for special tokens -> invalid with parent token
            probs[0] = 0  # 'padding' token

            # sample next sequence token
            target[cur_token] = torch.multinomial(probs, num_samples=1)[0]  # TODO: check input_len == i case.

            # update counter
            remaining_tokens -= 1
            cur_token += 1

    return value


def postprocess_iterative(value, target_resolution, spatial_dim, **_):
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
