import torch

from .basic_generator import BasicGenerator


class CompositeGenerator(BasicGenerator):
    def __init__(self, compute_logits_fn, num_tokens=[1], **_):
        """ Create token generator instance for a 'basic' head.

        Args:
            compute_logits_fn: Pointer to function, which computes logits of given sequence.
            num_tokens: Defines the number of sampled tokens in each step.
        """
        super(CompositeGenerator, self).__init__(compute_logits_fn, num_tokens[0])
        self.num_tokens_list = num_tokens

    def __call__(self, val, dep, pos, memory=None, layer_idx=0, start_idx=0, temperature=1.0, **_):
        """ Sample autoregressively current value token sequence and return updated value sequence.

        Args:
            val: Value token sequence of current layer.
            dep: Depth token sequence of current layer.
            pos: Position token sequence of current layer.
            memory: Latent sequence vector of the previous layer.
            layer_idx: Currently sampled layer index.
            start_idx: Start sampling with this token idx.
            temperature: Defines the randomness of the samples.

        Return:
            Sampled token sequence with values of the current layer.
        """
        # get the currently sampled depth
        cur_depth = torch.max(dep[-1])
        # assign number of sampled tokens accordingly to depth
        self.num_tokens = self.num_tokens_list[cur_depth - 1]
        # call the parent class sampler
        return super(CompositeGenerator, self).__call__(
            val,
            dep,
            pos,
            memory,
            layer_idx,
            start_idx,
            temperature,
            cur_depth < 6,  # check for substitution
        )
