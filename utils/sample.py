import torch
from tqdm.auto import tqdm
from utils.sample_utils import append_next_layer_tokens


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
