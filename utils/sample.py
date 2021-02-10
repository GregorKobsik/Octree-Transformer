import torch
from tqdm.auto import tqdm


def append_next_layer_tokens(seq, depth, pos_x, pos_y):
    # got an empty input - initialize with default values and return
    if len(seq) == 0:
        seq = torch.tensor([0], device=seq.device, dtype=torch.long)
        depth = torch.tensor([1], device=seq.device, dtype=torch.long)
        pos_x = torch.tensor([32], device=seq.device, dtype=torch.long)
        pos_y = torch.tensor([32], device=seq.device, dtype=torch.long)
        return seq, depth, pos_x, pos_y, 1

    # compute next layer depth and number of future tokens
    max_depth = torch.max(depth)
    num_future_tokens = 4 * torch.sum(seq[depth == max_depth] == 2)

    # compute future sequence (as padding) and future depth sequence
    next_layer_seq = torch.tensor(num_future_tokens * [0], device=seq.device, dtype=torch.long)
    next_layer_depth = torch.tensor(num_future_tokens * [max_depth + 1], device=seq.device, dtype=torch.long)

    # compute futue positions sequence
    pos_step = pos_x[0] // 2**max_depth
    next_layer_pos_x = torch.tensor([], device=seq.device, dtype=torch.long)
    for x in pos_x[torch.logical_and(seq == 2, depth == max_depth)]:
        next_pos_x = torch.tensor(
            [x - pos_step, x + pos_step, x - pos_step, x + pos_step], device=seq.device, dtype=torch.long
        )
        next_layer_pos_x = torch.cat([next_layer_pos_x, next_pos_x.long()])

    pos_step = pos_y[0] // 2**max_depth
    next_layer_pos_y = torch.tensor([], device=seq.device, dtype=torch.long)
    for y in pos_y[torch.logical_and(seq == 2, depth == max_depth)]:
        next_pos_y = torch.tensor(
            [y - pos_step, y - pos_step, y + pos_step, y + pos_step], device=seq.device, dtype=torch.long
        )
        next_layer_pos_y = torch.cat([next_layer_pos_y, next_pos_y.long()])

    # concat future tokens and return
    seq = torch.cat([seq, next_layer_seq])
    depth = torch.cat([depth, next_layer_depth])
    pos_x = torch.cat([pos_x, next_layer_pos_x])
    pos_y = torch.cat([pos_y, next_layer_pos_y])
    return seq, depth, pos_x, pos_y, num_future_tokens


def sample_sequence(model, seq, depth, pos_x, pos_y, max_len, max_depth, temperature=1.0):
    remaining_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(len(seq), max_len), leave=False, desc="Sampling"):
            # preprocess token sequences
            if remaining_tokens == 0:
                # depth set to maximum of 6
                if max(depth) == max_depth:
                    break
                # append future depth and position tokens to sequences
                seq, depth, pos_x, pos_y, remaining_tokens = append_next_layer_tokens(seq, depth, pos_x, pos_y)

            # compute logits of next token
            logits = model(
                seq[:i + 1].unsqueeze(-1),
                depth[:i + 1].unsqueeze(-1),
                torch.stack([pos_x[:i + 1], pos_y[:i + 1]]).unsqueeze(-1),
            )

            # sample next sequence token
            logits = logits[-1, :, :] / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            probs[0][0] = 0  # do not sample 'padding' tokens.
            pred = torch.multinomial(probs, num_samples=1)

            # update sequence
            seq[i] = pred[0]
            remaining_tokens -= 1

    return seq
