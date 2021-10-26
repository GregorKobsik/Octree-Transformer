import torch

from .basic_generator import BasicGenerator
from .substitution_generator import SubstitutionGenerator
from .double_substitution_generator import DoubleSubstitutionGenerator


class CompositeGenerator():
    def __init__(self, compute_logits_fn, num_tokens=[1], **_):
        """ Create token generator instance for a 'basic' head.

        Args:
            compute_logits_fn: Pointer to function, which computes logits of given sequence.
            num_tokens: Defines the number of sampled tokens in each step.
        """
        self.compute_logits_fn = compute_logits_fn
        self.num_tokens_list = num_tokens

    def __call__(self, val, dep, pos, memory=None, layer_idx=0, temperature=1.0, cls=None, **_):
        """ Sample autoregressively current value token sequence and return sampled value sequence.

        Args:
            val: Value token sequence of previous and current layers as a list.
            dep: Depth token sequence of previous and current layers as a list.
            pos: Position token sequence of previous and current layers as a list.
            memory: Latent sequence vector of the previous layer.
            layer_idx: Currently sampled layer index.
            temperature: Defines the randomness of the samples.
            cls: class label for conditional generation.

        Return:
            Sampled token sequence with values of the current layer.
        """
        # get the currently sampled depth
        cur_depth = torch.max(dep[-1])
        # get number of sampled tokens accordingly to depth
        num_tokens = self.num_tokens_list[cur_depth - 1]
        # create a generator according to layer depth
        if cur_depth < 6:
            generator = BasicGenerator(self.compute_logits_fn, num_tokens)
        elif cur_depth == 6:  # 'substitution'
            generator = SubstitutionGenerator(self.compute_logits_fn, num_tokens)
        else:  # 'double_substitution'
            generator = DoubleSubstitutionGenerator(self.compute_logits_fn, num_tokens)
        # sample a single layer
        return generator(val, dep, pos, memory, layer_idx, temperature, cls=cls)
