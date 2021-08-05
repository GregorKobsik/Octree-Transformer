import unittest
import numpy as np
import numpy.testing as np_test

from utils import (TrinaryRepresentation, load_chair, load_airplane, quick_linearise, kdTree)


class TestConversionBasicTrinaryToken(unittest.TestCase):
    def test_dec_to_tri_dim2(self):
        repr_trans = TrinaryRepresentation(spatial_dim=2)

        repr = repr_trans.dec_to_tri(1)
        self.assertEqual(repr, [1, 1, 1, 1])

        repr = repr_trans.dec_to_tri(2)
        self.assertEqual(repr, [1, 1, 1, 2])

        repr = repr_trans.dec_to_tri(3)
        self.assertEqual(repr, [1, 1, 1, 3])

        repr = repr_trans.dec_to_tri(4)
        self.assertEqual(repr, [1, 1, 2, 1])

        repr = repr_trans.dec_to_tri(5)
        self.assertEqual(repr, [1, 1, 2, 2])

        repr = repr_trans.dec_to_tri(6)
        self.assertEqual(repr, [1, 1, 2, 3])

        repr = repr_trans.dec_to_tri(7)
        self.assertEqual(repr, [1, 1, 3, 1])

        repr = repr_trans.dec_to_tri(8)
        self.assertEqual(repr, [1, 1, 3, 2])

        repr = repr_trans.dec_to_tri(9)
        self.assertEqual(repr, [1, 1, 3, 3])

        repr = repr_trans.dec_to_tri(10)
        self.assertEqual(repr, [1, 2, 1, 1])

        repr = repr_trans.dec_to_tri(20)
        self.assertEqual(repr, [1, 3, 1, 2])

        repr = repr_trans.dec_to_tri(30)
        self.assertEqual(repr, [2, 1, 1, 3])

        repr = repr_trans.dec_to_tri(43)
        self.assertEqual(repr, [2, 2, 3, 1])

        repr = repr_trans.dec_to_tri(81)
        self.assertEqual(repr, [3, 3, 3, 3])

    def test_tri_to_dec_dim2(self):
        repr_trans = TrinaryRepresentation(spatial_dim=3)

        repr = repr_trans.tri_to_dec([1, 1, 1, 1])
        self.assertEqual(repr, 1)

        repr = repr_trans.tri_to_dec([1, 1, 1, 2])
        self.assertEqual(repr, 2)

        repr = repr_trans.tri_to_dec([1, 1, 1, 3])
        self.assertEqual(repr, 3)

        repr = repr_trans.tri_to_dec([1, 1, 2, 1])
        self.assertEqual(repr, 4)

        repr = repr_trans.tri_to_dec([1, 1, 2, 2])
        self.assertEqual(repr, 5)

        repr = repr_trans.tri_to_dec([1, 1, 2, 3])
        self.assertEqual(repr, 6)

        repr = repr_trans.tri_to_dec([1, 1, 3, 1])
        self.assertEqual(repr, 7)

        repr = repr_trans.tri_to_dec([1, 1, 3, 2])
        self.assertEqual(repr, 8)

        repr = repr_trans.tri_to_dec([1, 1, 3, 3])
        self.assertEqual(repr, 9)

        repr = repr_trans.tri_to_dec([1, 2, 1, 1])
        self.assertEqual(repr, 10)

        repr = repr_trans.tri_to_dec([1, 3, 1, 2])
        self.assertEqual(repr, 20)

        repr = repr_trans.tri_to_dec([2, 1, 1, 3])
        self.assertEqual(repr, 30)

        repr = repr_trans.tri_to_dec([2, 2, 3, 1])
        self.assertEqual(repr, 43)

        repr = repr_trans.tri_to_dec([3, 3, 3, 3])
        self.assertEqual(repr, 81)

    def test_dec_to_tri_dim3(self):
        repr_trans = TrinaryRepresentation(spatial_dim=3)

        repr = repr_trans.dec_to_tri(1)
        self.assertEqual(repr, [1, 1, 1, 1, 1, 1, 1, 1])

        repr = repr_trans.dec_to_tri(2)
        self.assertEqual(repr, [1, 1, 1, 1, 1, 1, 1, 2])

        repr = repr_trans.dec_to_tri(3)
        self.assertEqual(repr, [1, 1, 1, 1, 1, 1, 1, 3])

        repr = repr_trans.dec_to_tri(4)
        self.assertEqual(repr, [1, 1, 1, 1, 1, 1, 2, 1])

        repr = repr_trans.dec_to_tri(5)
        self.assertEqual(repr, [1, 1, 1, 1, 1, 1, 2, 2])

        repr = repr_trans.dec_to_tri(6)
        self.assertEqual(repr, [1, 1, 1, 1, 1, 1, 2, 3])

        repr = repr_trans.dec_to_tri(7)
        self.assertEqual(repr, [1, 1, 1, 1, 1, 1, 3, 1])

        repr = repr_trans.dec_to_tri(8)
        self.assertEqual(repr, [1, 1, 1, 1, 1, 1, 3, 2])

        repr = repr_trans.dec_to_tri(9)
        self.assertEqual(repr, [1, 1, 1, 1, 1, 1, 3, 3])

        repr = repr_trans.dec_to_tri(10)
        self.assertEqual(repr, [1, 1, 1, 1, 1, 2, 1, 1])

        repr = repr_trans.dec_to_tri(100)
        self.assertEqual(repr, [1, 1, 1, 2, 1, 3, 1, 1])

        repr = repr_trans.dec_to_tri(1000)
        self.assertEqual(repr, [1, 2, 2, 1, 2, 1, 1, 1])

        repr = repr_trans.dec_to_tri(2853)
        self.assertEqual(repr, [2, 1, 3, 3, 1, 2, 3, 3])

        repr = repr_trans.dec_to_tri(6561)
        self.assertEqual(repr, [3, 3, 3, 3, 3, 3, 3, 3])

    def test_tri_to_dec_dim3(self):
        repr_trans = TrinaryRepresentation(spatial_dim=3)

        repr = repr_trans.tri_to_dec([1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(repr, 1)

        repr = repr_trans.tri_to_dec([1, 1, 1, 1, 1, 1, 1, 2])
        self.assertEqual(repr, 2)

        repr = repr_trans.tri_to_dec([1, 1, 1, 1, 1, 1, 1, 3])
        self.assertEqual(repr, 3)

        repr = repr_trans.tri_to_dec([1, 1, 1, 1, 1, 1, 2, 1])
        self.assertEqual(repr, 4)

        repr = repr_trans.tri_to_dec([1, 1, 1, 1, 1, 1, 2, 2])
        self.assertEqual(repr, 5)

        repr = repr_trans.tri_to_dec([1, 1, 1, 1, 1, 1, 2, 3])
        self.assertEqual(repr, 6)

        repr = repr_trans.tri_to_dec([1, 1, 1, 1, 1, 1, 3, 1])
        self.assertEqual(repr, 7)

        repr = repr_trans.tri_to_dec([1, 1, 1, 1, 1, 1, 3, 2])
        self.assertEqual(repr, 8)

        repr = repr_trans.tri_to_dec([1, 1, 1, 1, 1, 1, 3, 3])
        self.assertEqual(repr, 9)

        repr = repr_trans.tri_to_dec([1, 1, 1, 1, 1, 2, 1, 1])
        self.assertEqual(repr, 10)

        repr = repr_trans.tri_to_dec([1, 1, 1, 2, 1, 3, 1, 1])
        self.assertEqual(repr, 100)

        repr = repr_trans.tri_to_dec([1, 2, 2, 1, 2, 1, 1, 1])
        self.assertEqual(repr, 1000)

        repr = repr_trans.tri_to_dec([2, 1, 3, 3, 1, 2, 3, 3])
        self.assertEqual(repr, 2853)

        repr = repr_trans.tri_to_dec([3, 3, 3, 3, 3, 3, 3, 3])
        self.assertEqual(repr, 6561)


class TestConversionBasicTrinarySequence(unittest.TestCase):
    def test_basic_to_trinary_dim2(self):
        repr_trans = TrinaryRepresentation(spatial_dim=2)

        in_value = np.array([3, 2, 2, 1] + [3, 1, 3, 1, 3, 1, 1, 1])
        in_depth = np.array([1, 1, 1, 1] + [2, 2, 2, 2, 2, 2, 2, 2])
        in_pos = np.array(
            [[2, 2], [2, 6], [6, 2], [6, 6]] + [[1, 5], [1, 7], [3, 5], [3, 7], [5, 1], [5, 3], [7, 1], [7, 3]]
        )

        tgt_value = np.array([67, 61, 55])
        tgt_depth = np.array([1, 2, 2])
        tgt_pos = np.array([[4, 4], [2, 6], [6, 2]])

        out_value, out_depth, out_pos = repr_trans.encode_trinary(in_value, in_depth, in_pos)

        np_test.assert_array_equal(out_value, tgt_value)
        np_test.assert_array_equal(out_depth, tgt_depth)
        np_test.assert_array_equal(out_pos, tgt_pos)

    def test_trinary_to_basic_dim2(self):
        repr_trans = TrinaryRepresentation(spatial_dim=2)

        in_value = np.array([67, 61, 55])
        in_depth = np.array([1, 2, 2])
        in_pos = np.array([[4, 4], [2, 6], [6, 2]])

        tgt_value = np.array([3, 2, 2, 1] + [3, 1, 3, 1, 3, 1, 1, 1])
        tgt_depth = np.array([1, 1, 1, 1] + [2, 2, 2, 2, 2, 2, 2, 2])
        tgt_pos = np.array(
            [[2, 2], [2, 6], [6, 2], [6, 6]] + [[1, 5], [1, 7], [3, 5], [3, 7], [5, 1], [5, 3], [7, 1], [7, 3]]
        )

        out_value, out_depth, out_pos = repr_trans.decode_trinary(in_value, in_depth, in_pos)

        np_test.assert_array_equal(out_value, tgt_value)
        np_test.assert_array_equal(out_depth, tgt_depth)
        np_test.assert_array_equal(out_pos, tgt_pos)


class TestQuickLinearise(unittest.TestCase):
    """ Tests the 'quick_linearise' function, iff the computed results are equal to the kdTree class.
    """

    def quick_linearisation(self, voxels: np.ndarray, pos_encoding: str):
        """ Provides a unified test setup function for different arguments.

        Args:
            voxels (np.ndarray): Numpy array holding a binarized shape.
            pos_encoding (str): Defines the position encoding.
        """
        sequence = quick_linearise(voxels, pos_encoding=pos_encoding)
        target = kdTree(spatial_dim=3, pos_encoding=pos_encoding).insert_element_array(voxels).get_token_sequence(
            return_depth=True, return_pos=True
        )

        np_test.assert_array_equal(sequence[0], target[0])
        np_test.assert_array_equal(sequence[1], target[1])
        np_test.assert_array_equal(sequence[2], target[2])

    def test_chair_centered(self):
        """ Tests 'quick_linearise' with a chair input and centered encoding.
        """
        voxels = load_chair("/clusterarchive/ShapeNet/voxelization", 16)
        self.quick_linearisation(voxels, pos_encoding="centered")

    def test_chair_intertwined(self):
        """ Tests 'quick_linearise' with a chair input and intertwined encoding.
        """
        voxels = load_chair("/clusterarchive/ShapeNet/voxelization", 16)
        self.quick_linearisation(voxels, pos_encoding="intertwined")

    def test_airplane_centered(self):
        """ Tests 'quick_linearise' with a airplane input and centered encoding.
        """
        voxels = load_airplane("/clusterarchive/ShapeNet/voxelization", 16)
        self.quick_linearisation(voxels, pos_encoding="centered")

    def test_airplane_intertwined(self):
        """ Tests 'quick_linearise' with a airplane input and intertwined encoding.
        """
        voxels = load_airplane("/clusterarchive/ShapeNet/voxelization", 16)
        self.quick_linearisation(voxels, pos_encoding="intertwined")
