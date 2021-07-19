import torch

from tqdm.auto import trange


class SubstitutionGenerator():
    def __init__(self, compute_logits_fn, num_tokens=64, **_):
        """ Create token generator instance which samples 'num_tokens' in one pass.

        Args:
            compute_logits_fn: Pointer to function, which computes logits of given sequence.
            num_tokens: Defines the number of sampled tokens in each step.
        """
        self.compute_logits = compute_logits_fn
        self.num_tokens = num_tokens

    def __call__(self, val, dep, pos, memory=None, layer_idx=0, start_idx=0, temperature=1.0, **_):
        """ Sample autoregressively current value token sequence and return updated value sequence.

        Args:
            val: Value token sequence of currently sampled layer.
            dep: Depth token sequence of currently sampled layer.
            pos: Position token sequence of currently sampled layer.
            memory: Latent sequence vector of the previous layer.
            layer_idx: Currently sampled layer index.
            start_idx: Start sampling with this token idx.
            temperature: Defines the randomness of the samples.

        Return:
            Sampled token sequence with values of the current layer.
        """
        # extract sequences
        stop_idx = len(val[-1])

        # sample tokens autoregressively
        for token_idx in trange(start_idx, stop_idx, self.num_tokens, leave=False, desc="Tokens"):
            # batch and pack token sequences
            seq = (
                torch.cat([val[-2], val[-1]]).unsqueeze(0),
                torch.cat([dep[-2], dep[-1]]).unsqueeze(0),
                torch.cat([pos[-2], pos[-1]]).unsqueeze(0),
            )

            # compute logits
            logits = self.compute_logits(seq, memory, layer_idx)[0]

            # retrive only logits for for current index
            sampled_token_logits = logits[token_idx:token_idx + self.num_tokens]

            # check transformer token capacity
            if len(sampled_token_logits) == 0:
                return val[-1][:token_idx]  # reached maximum number of tokens

            # compute token probabilities from logits
            probs = torch.nn.functional.softmax(sampled_token_logits / temperature, dim=-1)  # [t, V]

            # zero probability for special tokens -> invalid with parent token
            probs[:, 0] = 0  # 'padding' token

            # sample next sequence token
            for i in range(len(probs)):
                val[-1][token_idx + i] = torch.multinomial(probs[i], num_samples=1)[0]

        return val[-1]
