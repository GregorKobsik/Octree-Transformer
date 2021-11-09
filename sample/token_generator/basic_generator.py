import torch

from tqdm.auto import trange


class BasicGenerator:
    def __init__(self, compute_logits_fn, num_tokens=1, **_):
        """ Create token generator instance which samples 'num_tokens' in one pass.

        Args:
            compute_logits_fn: Pointer to function, which computes logits of given sequence.
            num_tokens: Defines the number of sampled tokens in each step.
        """
        self.compute_logits = compute_logits_fn
        self.kernel_size = num_tokens

    def __call__(self, val, dep, pos, memory=None, idx=0, temperature=1.0, slice_sequence=True, cls=None, **_):
        """ Sample autoregressive current value token sequence and return updated value sequence.

        Args:
            val: Value token sequence of current layer.
            dep: Depth token sequence of current layer.
            pos: Position token sequence of current layer.
            memory: Latent sequence vector of the previous layer.
            idx: Currently sampled transformer layer index.
            temperature: Defines the randomness of the samples.
            cls: class label for conditional generation.

        Return:
            Sampled token sequence with values of the current layer.
        """
        # compute indices
        token_idx = 0
        sampled_idx = len(torch.cat(val[:-1])) if len(val) > 1 else 0

        # sample tokens autoregressive
        for _ in trange(len(val[-1]) // self.kernel_size, leave=False, desc="Tokens"):

            for block_idx in range(self.kernel_size):
                # concat layers and slice sequence for speed_up
                seq = (
                    torch.cat(val)[:sampled_idx + token_idx + self.kernel_size].unsqueeze(0),
                    torch.cat(dep)[:sampled_idx + token_idx + self.kernel_size].unsqueeze(0),
                    torch.cat(pos)[:sampled_idx + token_idx + self.kernel_size].unsqueeze(0),
                )

                logits = self.compute_logits(seq, memory, idx, cls)[0]

                # retrieve only logits for for current index
                sampled_token_logits = logits[sampled_idx + token_idx + block_idx]

                # compute token probabilities from logits
                sampled_token_logits[0] = -float("Inf")  # 'padding' token
                probs = torch.nn.functional.softmax(sampled_token_logits / temperature, dim=-1)  # [t, V]

                # sample next sequence token
                val[-1][token_idx + block_idx] = torch.multinomial(probs, num_samples=1)[0]

            # update indices
            token_idx += self.kernel_size

        return val[-1]
