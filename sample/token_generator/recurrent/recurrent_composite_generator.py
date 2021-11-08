import torch

from .recurrent_basic_generator import RecurrentBasicGeneratorAutoregressive
from .recurrent_substitution_generator import RecurrentSubstitutionGeneratorAutoregressive


class RecurrentCompositeGeneratorAutoregressive:
    def __init__(self, embed_fn, transformer_fn, head_fn, num_tokens=[1], **_):
        """ Create token generator instance for a 'basic' head.

        Args:
            embed_fn: Pointer to function, which processes the token embedding of the Shape Transformer.
            transformer_fn: Pointer to function, which processes the Transformer module of the Shape Transformer.
            head_fn: Pointer to function, which processes the generative head of the Shape Transformer.
            num_tokens: Defines the number of sampled tokens in each step for each single depth layer.
        """
        self.model_fn = {
            'embed_fn': embed_fn,
            'transformer_fn': transformer_fn,
            'head_fn': head_fn,
        }
        self.num_tokens_list = num_tokens

    def __call__(self, val, dep, pos, memory=None, state=None, temperature=1.0, cls=None, **_):
        """ Sample autoregressively current value token sequence and return sampled value sequence.

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
        # get the currently sampled depth
        cur_depth = torch.max(dep[-1])
        # get number of sampled tokens accordingly to depth
        num_tokens = self.num_tokens_list[cur_depth - 1]
        # create a generator according to layer depth
        if cur_depth < 6:
            generator = RecurrentBasicGeneratorAutoregressive(num_tokens=num_tokens, **self.model_fn)
        elif cur_depth == 6:  # 'substitution'
            generator = RecurrentSubstitutionGeneratorAutoregressive(num_tokens=num_tokens, **self.model_fn)
        # else:  # 'double_substitution'
        #     generator = RecurrentDoubleSubstitutionGeneratorAutoregressive(num_tokens=num_tokens, **self.model_fn)
        # sample a single layer
        return generator(val, dep, pos, memory, state, temperature, cls=cls)
