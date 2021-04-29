import torch
from tqdm.auto import tqdm

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


def sample_sequence(model, value, depth, pos, spatial_dim, max_len, max_depth, batch_first=False, temperature=1.0):
    remaining_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(len(value), max_len), leave=False, desc="Sampling"):
            # preprocess token sequences
            if remaining_tokens == 0:
                # depth set to maximum of 6
                if max(depth) == max_depth:
                    break
                # append future depth and position tokens to sequences
                value, depth, pos, remaining_tokens = append_next_layer_tokens(value, depth, pos, spatial_dim)
                # all tokens are final, return sequence
                if remaining_tokens == 0:
                    return value

            # compute logits of next token
            logits = model(
                value[:i + 1].unsqueeze(1 - batch_first),  # [N, S] or [S, N]
                depth[:i + 1].unsqueeze(1 - batch_first),  # [N, S] or [S, N]
                pos[:, :i + 1].unsqueeze(2 - batch_first),  # [A, N, S] or [A, S, N]
            )

            # sample next sequence token
            logits = logits[:, -1, :] if batch_first else logits[-1, :, :]
            logits /= temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            probs[0][0] = 0  # do not sample 'padding' tokens.
            pred = torch.multinomial(probs, num_samples=1)

            # debugging
            if pred[0] not in [1, 2, 3]:
                print("probs:", probs)

            # update sequence
            if (len(value) == i):
                return value
            value[i] = pred[0]
            remaining_tokens -= 1

    return value
