import torch

from tqdm.auto import trange


class SubstitutionGenerator:
    def __init__(self, compute_logits_fn, num_tokens=8, **_):
        """ Create token generator instance which samples 'num_tokens' in one pass.

        Args:
            compute_logits_fn: Pointer to function, which computes logits of given sequence.
            num_tokens: Defines the number of sampled tokens in each step.
        """
        self.compute_logits = compute_logits_fn
        self.kernel_size = num_tokens

    def __call__(self, val, dep, pos, memory=None, idx=0, temperature=1.0, cls=None, **_):
        """ Sample autoregressive current value token sequence and return updated value sequence.

        Args:
            val: Value token sequence of currently sampled layer.
            dep: Depth token sequence of currently sampled layer.
            pos: Position token sequence of currently sampled layer.
            memory: Latent sequence vector of the previous layer.
            idx: Currently sampled transformer layer index.
            temperature: Defines the randomness of the samples.

        Return:
            Sampled token sequence with values of the current layer.
        """
        # compute indices
        token_idx = 0
        second_last_idx = 0
        sampled_idx = len(torch.cat(val[:-1])) if len(val) > 2 else 0

        # sample tokens autoregressive
        for _ in trange(len(val[-2]) // self.kernel_size, leave=False, desc="Tokens"):
            # compute number of tokens which can be sampled
            mix_second_last = torch.sum(val[-2][second_last_idx:second_last_idx + self.kernel_size] == 2)
            num_sampled = mix_second_last * 8

            for block_idx in range(num_sampled.item()):
                # concat and pack token sequences to compute logits
                seq = (torch.cat(val).unsqueeze(0), torch.cat(dep).unsqueeze(0), torch.cat(pos).unsqueeze(0))

                logits = self.compute_logits(seq, memory, idx, cls)[0]

                # retrieve only logits for for current index
                sampled_token_logits = logits[sampled_idx + token_idx + block_idx]

                # compute token probabilities from logits
                sampled_token_logits[0] = -float("Inf")  # 'padding' token
                probs = torch.nn.functional.softmax(sampled_token_logits / temperature, dim=-1)  # [t, V]

                # sample next sequence token
                val[-1][token_idx + block_idx] = torch.multinomial(probs, num_samples=1)[0]

            # update indices
            second_last_idx += self.kernel_size
            token_idx += num_sampled

        return val[-1]
