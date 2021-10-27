import unittest
import numpy as np

from data.transform import create_data_transform


class TestSequenceLengthTransform(unittest.TestCase):
    """ Test the SequenceLengthTransform class """
    def test_basic_embedding(self):
        """ Test a small basic embedding example.

        The first comparison should return exactly the input, as the sequence is within the limit.
        The second comparison should return `None`, as the input exceeds the token limit.
        """
        val = np.array([3, 2, 2, 1, 1, 1, 3, 1, 2, 3, 1, 2, 1, 3, 1, 1, 1, 3, 1, 1])
        dep = np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3])
        pos = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # unused
        seq = (val, dep, pos)

        transform_valid = create_data_transform(
            name="check_len",
            spatial_dim=3,
            resolution=32,
            position_encoding='centered',
            num_positions=20,
            embedding=['basic'],
        )
        transform_invalid = create_data_transform(
            name="check_len",
            spatial_dim=3,
            resolution=32,
            position_encoding='centered',
            num_positions=19,
            embedding=['basic'],
        )

        self.assertEqual(seq, transform_valid(seq))
        self.assertEqual(None, transform_invalid(seq))

    def test_convolution_embedding(self):
        """ Test a small convolutional embedding example.

        The first comparison should return exactly the input, as the sequence is within the limit.
        The second comparison should return `None`, as the input exceeds the token limit.
        """
        val = np.array(8 * [1] + 8 * [2, 2, 2] + 8 * [3, 3, 3, 3, 3, 3, 3, 3])
        dep = np.array(8 * [1] + 8 * [2, 2, 2] + 8 * [3, 3, 3, 3, 3, 3, 3, 3])
        pos = np.array(8 * [1] + 8 * [2, 2, 2] + 8 * [3, 3, 3, 3, 3, 3, 3, 3])  # unused
        seq = (val, dep, pos)

        transform_valid = create_data_transform(
            name="check_len",
            spatial_dim=3,
            resolution=32,
            position_encoding='centered',
            num_positions=12,
            embedding=['single_conv'],
        )
        transform_invalid = create_data_transform(
            name="check_len",
            spatial_dim=3,
            resolution=32,
            position_encoding='centered',
            num_positions=11,
            embedding=['single_conv'],
        )

        self.assertEqual(seq, transform_valid(seq))
        self.assertEqual(None, transform_invalid(seq))

    def test_composite_embedding_small(self):
        """ Test a small composite embedding example. All embeddings in the convolution are 'basic'.

        The first comparison should return exactly the input, as the sequence is within the limit.
        The second comparison should return `None`, as the input exceeds the token limit.
        """
        val = np.array(8 * [1] + 8 * [2, 2] + 8 * [3, 3])
        dep = np.array(8 * [1] + 8 * [2, 2] + 8 * [3, 3])
        pos = np.array(8 * [1] + 8 * [2, 2] + 8 * [3, 3])  # unused
        seq = (val, dep, pos)

        transform_valid = create_data_transform(
            name="check_len",
            spatial_dim=3,
            resolution=32,
            position_encoding='centered',
            num_positions=40,
            embedding=['composite'],
        )
        transform_invalid = create_data_transform(
            name="check_len",
            spatial_dim=3,
            resolution=32,
            position_encoding='centered',
            num_positions=39,
            embedding=['composite'],
        )

        self.assertEqual(seq, transform_valid(seq))
        self.assertEqual(None, transform_invalid(seq))

    def test_composite_embedding_medium(self):
        """ Test a small composite embedding example. Uses 'basic', 'half_conv' and 'single_conv' embeddings.

        The first comparison should return exactly the input, as the sequence is within the limit.
        The second comparison should return `None`, as the input exceeds the token limit.
        """
        val = np.array(8 * [1] + 8 * [2] + 8 * [3] + 8 * [4] + 8 * [5, 5, 5, 5, 5])
        dep = np.array(8 * [1] + 8 * [2] + 8 * [3] + 8 * [4] + 8 * [5, 5, 5, 5, 5])
        pos = np.array(8 * [1] + 8 * [2] + 8 * [3] + 8 * [4] + 8 * [5, 5, 5, 5, 5])
        seq = (val, dep, pos)

        transform_valid = create_data_transform(
            name="check_len",
            spatial_dim=3,
            resolution=32,
            position_encoding='centered',
            num_positions=31,
            embedding=['composite'],
        )
        transform_invalid = create_data_transform(
            name="check_len",
            spatial_dim=3,
            resolution=32,
            position_encoding='centered',
            num_positions=30,
            embedding=['composite'],
        )

        self.assertEqual(seq, transform_valid(seq))
        self.assertEqual(None, transform_invalid(seq))

    def test_composite_embedding_large(self):
        """ Test a small composite embedding example. Uses all embedding types.

        The first comparison should return exactly the input, as the sequence is within the limit.
        The second comparison should return `None`, as the input exceeds the token limit.
        """
        val = np.array(8 * [1] + 8 * [2] + 8 * [3] + 8 * [4] + 8 * [5, 5, 5, 5, 5] + 8 * [6, 6, 6] + 8 * [7, 7, 7])
        dep = np.array(8 * [1] + 8 * [2] + 8 * [3] + 8 * [4] + 8 * [5, 5, 5, 5, 5] + 8 * [6, 6, 6] + 8 * [7, 7, 7])
        pos = np.array(8 * [1] + 8 * [2] + 8 * [3] + 8 * [4] + 8 * [5, 5, 5, 5, 5] + 8 * [6, 6, 6] + 8 * [7, 7, 7])
        seq = (val, dep, pos)

        transform_valid = create_data_transform(
            name="check_len",
            spatial_dim=3,
            resolution=32,
            position_encoding='centered',
            num_positions=41,
            embedding=['composite'],
        )
        transform_invalid = create_data_transform(
            name="check_len",
            spatial_dim=3,
            resolution=32,
            position_encoding='centered',
            num_positions=40,
            embedding=['composite'],
        )

        self.assertEqual(seq, transform_valid(seq))
        self.assertEqual(None, transform_invalid(seq))
