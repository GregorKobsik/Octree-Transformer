import torch

from tqdm.auto import trange


class RecurrentSubstitutionGeneratorAutoregressive:
    def __init__(self, embed_fn, transformer_fn, head_fn, num_tokens=8, **_):
        """ Create token generator instance which samples 'num_tokens' in one pass.

        Args:
            embed_fn: Pointer to function, which processes the token embedding of the Shape Transformer.
            transformer_fn: Pointer to function, which processes the Transformer module of the Shape Transformer.
            head_fn: Pointer to function, which processes the generative head of the Shape Transformer.
            num_tokens: Defines the number of sampled tokens in each step.
        """
        self.embed_fn = embed_fn
        self.transformer_fn = transformer_fn
        self.head_fn = head_fn
        self.kernel_size = num_tokens

    def __call__(self, val, dep, pos, memory=None, state=None, temperature=1.0, cls=None, **_):
        """ Sample autoregressively current value token sequence and return updated value sequence.

        Args:
            val: Value token sequence of current layer.
            dep: Depth token sequence of current layer.
            pos: Position token sequence of current layer.
            memory: Latent sequence vector of the previous layers.
            state: Internal state object of the Transformer
            temperature: Defines the randomness of the samples.
            cls: class label for conditional generation.

        Return:
            Sampled token sequence with values of the current layer.
        """
        # init indices
        token_idx = 0
        second_last_idx = 0
        memory_idx = len(memory[0]) if memory is not None else 0

        # sample tokens autoregressively
        for idx in trange(0, len(val[-2]) // self.kernel_size, leave=False, desc="Tokens"):
            # compute number of tokens which can be sampled
            mix_second_last = torch.sum(val[-2][second_last_idx:second_last_idx + self.kernel_size] == 2)
            num_sampled = mix_second_last * 8

            # embed sequence
            seq = (torch.cat(val).unsqueeze(0), torch.cat(dep).unsqueeze(0), torch.cat(pos).unsqueeze(0))
            input_token = self.embed_fn(seq, cls)[:, memory_idx + idx]

            # process a single token with the Transformer and append output to memory sequence
            out, state = self.transformer_fn(input_token, state)
            memory = torch.cat((memory, out.unsqueeze(0)), dim=1) if memory is not None else out.unsqueeze(0)

            # use an autoregressive head within the substitution block
            for block_idx in range(num_sampled.item()):
                # extract only a subsequence of seq and memory, which is actually used (+1 for lookahead embedding)
                seq = (
                    torch.cat(
                        (
                            val[-2][second_last_idx:second_last_idx + self.kernel_size],
                            val[-1][token_idx:token_idx + num_sampled + 1],
                        )
                    ).unsqueeze(0),
                    torch.cat(
                        (
                            dep[-2][second_last_idx:second_last_idx + self.kernel_size],
                            dep[-1][token_idx:token_idx + num_sampled + 1],
                        )
                    ).unsqueeze(0),
                    torch.cat(
                        (
                            pos[-2][second_last_idx:second_last_idx + self.kernel_size],
                            pos[-1][token_idx:token_idx + num_sampled + 1],
                        )
                    ).unsqueeze(0),
                )

                # compute logits from the memory vector and retrieve them for the current index only
                logits = self.head_fn(out.unsqueeze(0), seq, last_only=True)[0]  # squeeze(dim=0)
                sampled_token_logits = logits[block_idx]

                # compute token probabilities from logits
                sampled_token_logits[0] = -float("Inf")  # 'padding' token
                probs = torch.nn.functional.softmax(sampled_token_logits / temperature, dim=-1)

                # sample next sequence token and update sequence values
                sampled_val_token = torch.multinomial(probs, num_samples=1)[0]
                val[-1][token_idx + block_idx] = sampled_val_token

            # update indices
            second_last_idx += self.kernel_size
            token_idx += num_sampled

        return val[-1], memory, state
