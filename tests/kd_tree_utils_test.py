import unittest
import numpy as np
import numpy.testing as np_test

from utils.kd_tree_utils import RepresentationTransformator


class TestConversionTrinaryDecimal(unittest.TestCase):
    def test_dec_to_tri_dim2(self):
        repr_trans = RepresentationTransformator(spatial_dim=2)

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
        repr_trans = RepresentationTransformator(spatial_dim=3)

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
        repr_trans = RepresentationTransformator(spatial_dim=3)

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
        repr_trans = RepresentationTransformator(spatial_dim=3)

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


class TestConversionSuccessiveIterative(unittest.TestCase):
    def test_successive_to_iterative_dim2(self):
        repr_trans = RepresentationTransformator(spatial_dim=2)

        in_value = np.array([2] + [3, 2, 2, 1] + [3, 1, 3, 1, 3, 1, 1, 1])
        in_depth = np.array([1] + [2, 2, 2, 2] + [3, 3, 3, 3, 3, 3, 3, 3])
        in_pos = np.array(
            [[4, 4]] + [[2, 2], [6, 2], [2, 6], [6, 6]] +
            [[5, 1], [5, 3], [7, 1], [7, 3], [1, 5], [1, 7], [3, 5], [3, 7]]
        )

        tgt_value = np.array([2] + [3, 2, 2, 1])
        tgt_depth = np.array([1] + [2, 2, 2, 2])
        tgt_pos = np.array([[4, 4]] + [[2, 2], [6, 2], [2, 6], [6, 6]])
        tgt_target = np.array([81, 61, 55, 1])

        out_value, out_depth, out_pos, out_target = repr_trans.successive_to_iterative(in_value, in_depth, in_pos)

        np_test.assert_array_equal(out_value, tgt_value)
        np_test.assert_array_equal(out_depth, tgt_depth)
        np_test.assert_array_equal(out_pos, tgt_pos)
        np_test.assert_array_equal(out_target, tgt_target)

    def test_iterative_to_successive_dim2(self):
        repr_trans = RepresentationTransformator(spatial_dim=2)

        in_value = np.array([2] + [3, 2, 2, 1])
        in_depth = np.array([1] + [2, 2, 2, 2])
        in_pos = np.array([[4, 4]] + [[2, 2], [6, 2], [2, 6], [6, 6]])
        in_target = np.array([81, 61, 55, 1])

        tgt_value = np.array([2] + [3, 2, 2, 1] + [3, 1, 3, 1, 3, 1, 1, 1])
        tgt_depth = np.array([1] + [2, 2, 2, 2] + [3, 3, 3, 3, 3, 3, 3, 3])
        tgt_pos = np.array(
            [[4, 4]] + [[2, 2], [6, 2], [2, 6], [6, 6]] +
            [[5, 1], [5, 3], [7, 1], [7, 3], [1, 5], [1, 7], [3, 5], [3, 7]]
        )

        out_value, out_depth, out_pos = repr_trans.iterative_to_successive(in_value, in_depth, in_pos, in_target)

        np_test.assert_array_equal(out_value, tgt_value)
        np_test.assert_array_equal(out_depth, tgt_depth)
        np_test.assert_array_equal(out_pos, tgt_pos)
