import torch
import math

from utils import (
    _directions,
    kdTree,
)


def next_layer_tokens(value, depth, pos, spatial_dim, max_resolution):
    """ Creates artificial tokens for the next layer of the value sequence, to match the predefined shape. Precomputes
    corresponding depth and position tokens of the sequence, too.

    Args:
        value: Value token sequence given as a pytorch tensor.
        depth: Depth token sequence given as a pytorch tensor.
        pos: Position token sequence given as a pytorch tensor.
        spatial_dim: The spatial dimensionality of the value sequence.
        max_resolution: The maximal resolution the corresponding model is trained for.

    Return:
        (value, depth, pos)
    """
    cur_device = value.device
    dirs = _directions(spatial_dim)
    num_children = 2**spatial_dim

    # got an empty input - initialize with default values and return
    if len(value) == 0:
        value = torch.tensor(num_children * [1], device=cur_device, dtype=torch.long)
        depth = torch.tensor(num_children * [1], device=cur_device, dtype=torch.long)
        pos = torch.ones(num_children, spatial_dim, device=cur_device, dtype=torch.long) * max_resolution
        return value, depth, pos

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

    return nl_value, nl_depth, nl_pos


def preprocess(precondition, precondition_resolution, spatial_dim, device):
    """ Transform input array elements into token sequences.

    Args:
        precondition: An array of elements (pixels/voxels) as an numpy array.
        precondition_resolution: Resolution, to which the input array will be downscaled and used as a precondition.
        spatial_dim: The spatial dimensionality of the array of elements.
        device: Device on which, the data should be stored. Either "cpu" or "cuda" (gpu-support).

    Return:
        PyTorch tensor consisting of token sequences: (value, depth, position).
    """
    # convert input array into token sequence
    tree = kdTree(spatial_dim)
    tree = tree.insert_element_array(precondition, max_depth=math.log2(precondition_resolution) + 1)
    value, depth, pos = tree.get_token_sequence(
        depth=math.log2(precondition_resolution), return_depth=True, return_pos=True
    )

    # TODO: define trinary transformation based on list of embeddings

    # convert sequence tokens to PyTorch as a long tensor
    value = torch.tensor(value, dtype=torch.long, device=device)
    depth = torch.tensor(depth, dtype=torch.long, device=device)
    pos = torch.tensor(pos, dtype=torch.long, device=device)

    return value, depth, pos


def postprocess(value, target_resolution, spatial_dim):
    """ Transform sequence of value tokens into an array of elements (voxels/pixels).

    Args:
        value: Value token sequence as a pytorch tensor.
        target_resolution: Resolution up to which an object should be sampled.
        spatial_dim: The spatial dimensionality of the array of elements.

    Return:
        An array of elements as a numpy array.
    """
    # move value sequence to the cpu and convert to numpy array
    value = value.cpu().numpy()

    # TODO: define trinary transformation based on list of embeddings

    # insert the sequence into a kd-tree
    tree = kdTree(spatial_dim).insert_token_sequence(
        value,
        resolution=target_resolution,
        autorepair_errors=True,
        silent=True,
    )

    # retrive pixels/voxels from the kd-tree
    return tree.get_element_array(mode="occupancy")
