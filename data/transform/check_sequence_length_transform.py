import torch


class CheckSequenceLenghtTransform():

    # TODO: make this maps actually properties of the embedding class or decouple them from this module
    _substitution_level_map = {
        'basic': 0,
        'basic_A': 0,
        'half_conv': 0,
        'half_conv_A': 0,
        'single_conv': 0,
        'single_conv_A': 0,
        'multi_conv_A': 0,
        'substitution': 1,  # invalid single module embedding
        'double_substitution': 2,  # invalid single module embedding
        'composite': [0, 0, 0, 0, 0, 1, 2, 2],
        'composite_A': [0, 0, 0, 0, 0, 1, 2, 2],
        'composite_B': [0, 0, 0, 0, 0, 1],
        'composite_C': [0, 0, 0, 0, 0, 1],
        'composite_D': [0, 0, 0, 0, 1, 1, 2, 2],
    }

    _convolution_factor_map = {
        'basic': 1,
        'basic_A': 1,
        'half_conv': 4,
        'half_conv_A': 4,
        'single_conv': 8,
        'single_conv_A': 8,
        'multi_conv_A': 64,
        'substitution': 8,
        'double_substitution': 8,
        'composite': [1, 1, 1, 4, 8, 8, 8, 8],
        'composite_A': [1, 1, 1, 4, 8, 8, 8, 8],
        'composite_B': [1, 1, 1, 1, 8, 8],
        'composite_C': [1, 1, 2, 4, 8, 4],
        'composite_D': [1, 1, 4, 8, 4, 8, 4, 8],
    }

    def __init__(self, num_positions, embedding):
        """ Transform module to check if the sequence length is within the given input token limit `num_positions`.

        Args:
            num_positions: Maximal length of processed input tokens for the shape transformer.
            embedding: Defines the used token embedding of the shape transformer.
        """
        self.num_positions = num_positions

        # TODO: accepts only DO-Transformer with only a single token embedding
        self.convolution_factor = self._convolution_factor_map[embedding[0]]
        self.substitution_level = self._substitution_level_map[embedding[0]]

    def check_single_embedding(self, val, dep, pos):
        """ Check the embedded sequence length given a single token embedding module. """
        # Note: substitution is not valid for a single token embedding
        sequence_length = len(val) // self.convolution_factor
        if sequence_length > self.num_positions:
            return None
        else:
            return val, dep, pos

    def check_composite_embedding(self, val, dep, pos):
        """ Check the embedded sequence length given a composite token embedding with multiple modules. """
        sum_sequence_length = 0
        max_depth = max(dep)
        for i in range(min(len(self.substitution_level), max_depth)):
            sub_diff = self.substitution_level[i]
            conv_fac = self.convolution_factor[i]
            dep_level = i + 1 - sub_diff

            if sub_diff == 0:
                num_vectors = torch.sum(torch.from_numpy(dep) == dep_level) // conv_fac
            elif sub_diff == 1:
                val_1 = torch.from_numpy(val)[torch.from_numpy(dep) == (dep_level - 1)]
                num_vectors = (val_1.view(-1, conv_fac) == 2).max(dim=-1)[0].sum()
            elif sub_diff == 2:
                val_1 = torch.from_numpy(val)[torch.from_numpy(dep) == (dep_level - 1)]
                val_2 = torch.from_numpy(val)[torch.from_numpy(dep) == (dep_level - 2)]
                mask_1 = (val_1.view(-1, 8) == 2).max(dim=-1)[0]
                mask_2 = torch.zeros_like(val_2, dtype=torch.bool)
                mask_2[val_2 == 2] = mask_1
                mask_2 = mask_2.view(-1, conv_fac).max(dim=-1)[0]
                num_vectors = mask_2.sum()
            else:
                print("ERROR: substitution factors bigger than 2 are not implemented")
                return None

            sum_sequence_length += num_vectors

        if sum_sequence_length > self.num_positions:
            return None
        else:
            return val, dep, pos

    def __call__(self, seq, **_):
        """ Transforms a single sample to trinary representation. """
        if self.num_positions <= 0:
            return seq  # no maximum sequence length for the Transformer

        val, dep, pos = seq
        if type(self.convolution_factor) is list:
            return self.check_composite_embedding(val, dep, pos)
        else:
            return self.check_single_embedding(val, dep, pos)
