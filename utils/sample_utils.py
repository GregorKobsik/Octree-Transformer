import torch

_dirs = {
    1:
        torch.tensor([
            [-1, +1],
        ]),
    2:
        torch.tensor([  # TODO: swap dim 0, after remodeling quadtree/octree
            [-1, +1, -1, +1],
            [-1, -1, +1, +1],
        ]),
    3:
        torch.tensor(
            [
                [-1, -1, -1, -1, +1, +1, +1, +1],
                [-1, -1, +1, +1, -1, -1, +1, +1],
                [-1, +1, -1, +1, -1, +1, -1, +1],
            ]
        ),
}


def append_next_layer_tokens(value, depth, pos, spatial_dim=3):
    # got an empty input - initialize with default values and return
    cur_device = value.device
    if len(value) == 0:
        value = torch.tensor([0], device=cur_device, dtype=torch.long)
        depth = torch.tensor([1], device=cur_device, dtype=torch.long)
        pos = torch.ones(spatial_dim, 1, device=cur_device, dtype=torch.long) * 32
        num_future_tokens = torch.ones(1, device=cur_device, dtype=torch.long)
        return value, depth, pos, num_future_tokens

    # compute next layer depth and number of future tokens
    next_depth = torch.max(depth)
    num_future_tokens = 2**spatial_dim * torch.sum(value[depth == next_depth] == 2)

    # compute future sequence (as padding) and future depth sequence
    next_layer_value = torch.tensor([0], device=cur_device, dtype=torch.long).repeat(num_future_tokens)
    next_layer_depth = torch.tensor([next_depth + 1], device=cur_device, dtype=torch.long).repeat(num_future_tokens)

    # retrive and copy mixed tokens positions
    pos_token = pos[:, torch.logical_and(value == 2, depth == next_depth)]
    next_layer_pos = torch.repeat_interleave(pos_token, 2**spatial_dim, dim=1)

    # compute position difference and add it to future positions with respect to predefined pattern
    pos_step = pos[0][0] // 2**next_depth  # assume same resolution for each dimension
    next_layer_pos = next_layer_pos + pos_step * _dirs[spatial_dim].to(cur_device).repeat(1, pos_token.shape[1])

    # concat future tokens and return
    value = torch.cat([value, next_layer_value])
    depth = torch.cat([depth, next_layer_depth])
    pos = torch.cat([pos, next_layer_pos], dim=1)

    return value, depth, pos, num_future_tokens
