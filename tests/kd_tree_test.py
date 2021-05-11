import unittest
import numpy as np
import numpy.testing as np_test

from utils.kd_tree import kdTree


class TestWhiteboxSplit(unittest.TestCase):
    """ Tests the split function for quadtrees and octrees.

    Performs each one test with a small and a bigger input data for a quadtree and an octree.
    The given data should be split in half along each dimension beginning with the last one.
    """
    def test_split_dim2(self):
        quadtree = kdTree(spatial_dim=2)

        input = np.array([
            [1, 2],
            [3, 4],
        ])

        target = np.array([
            [[1]],
            [[2]],
            [[3]],
            [[4]],
        ])

        output = quadtree._split(input)

        self.assertEqual(input.shape, (2, 2))
        self.assertEqual(output.shape, (4, 1, 1))
        np_test.assert_array_equal(output, target)

        input = np.array([
            [11, 12, 13, 14],
            [15, 16, 17, 18],
            [19, 20, 21, 22],
            [23, 24, 25, 26],
        ])

        target = np.array([
            [[11, 12], [15, 16]],
            [[13, 14], [17, 18]],
            [[19, 20], [23, 24]],
            [[21, 22], [25, 26]],
        ])

        output = quadtree._split(input)

        self.assertEqual(input.shape, (4, 4))
        self.assertEqual(output.shape, (4, 2, 2))
        np_test.assert_array_equal(output, target)

    def test_split_dim3(self):
        octree = kdTree(spatial_dim=3)

        input = np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ])

        target = np.array([
            [[[1]]],
            [[[2]]],
            [[[3]]],
            [[[4]]],
            [[[5]]],
            [[[6]]],
            [[[7]]],
            [[[8]]],
        ])

        output = octree._split(input)

        self.assertEqual(input.shape, (2, 2, 2))
        self.assertEqual(output.shape, (8, 1, 1, 1))
        np_test.assert_array_equal(output, target)

        input = np.array(
            [
                [[11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22], [23, 24, 25, 26]],
                [[27, 28, 29, 30], [31, 32, 33, 34], [35, 36, 37, 38], [39, 40, 41, 42]],
                [[43, 44, 45, 46], [47, 48, 49, 50], [51, 52, 53, 54], [55, 56, 57, 58]],
                [[59, 60, 61, 62], [63, 64, 65, 66], [67, 68, 69, 70], [71, 72, 73, 74]],
            ]
        )

        target = np.array(
            [
                [[[11, 12], [15, 16]], [[27, 28], [31, 32]]],
                [[[13, 14], [17, 18]], [[29, 30], [33, 34]]],
                [[[19, 20], [23, 24]], [[35, 36], [39, 40]]],
                [[[21, 22], [25, 26]], [[37, 38], [41, 42]]],
                [[[43, 44], [47, 48]], [[59, 60], [63, 64]]],
                [[[45, 46], [49, 50]], [[61, 62], [65, 66]]],
                [[[51, 52], [55, 56]], [[67, 68], [71, 72]]],
                [[[53, 54], [57, 58]], [[69, 70], [73, 74]]],
            ]
        )

        output = octree._split(input)

        self.assertEqual(input.shape, (4, 4, 4))
        self.assertEqual(target.shape, (8, 2, 2, 2))
        np_test.assert_array_equal(output, target)


class TestWhiteboxConcat(unittest.TestCase):
    """ Tests the concatination function for quadtrees and octrees.

    Performs each one test with a small and a bigger input data for a quadtree and an octree.
    The given data should be concatinated along the first dimension in an interviened manner.

    The axes should be concatinated iteratively beginning with the second dimension, which
    is first splitted in half and than stacked together. The same is true for all remaining dimensions.
    """
    def test_concat_dim2(self):
        quadtree = kdTree(spatial_dim=2)

        input = np.array([
            [[1]],
            [[2]],
            [[3]],
            [[4]],
        ])

        target = np.array([
            [1, 2],
            [3, 4],
        ])

        output = quadtree._concat(input)
        self.assertEqual(input.shape, (4, 1, 1))
        self.assertEqual(output.shape, (2, 2))
        np_test.assert_array_equal(output, target)

        input = np.array([
            [[11, 12], [13, 14]],
            [[15, 16], [17, 18]],
            [[19, 20], [21, 22]],
            [[23, 24], [25, 26]],
        ])

        target = np.array([
            [11, 12, 15, 16],
            [13, 14, 17, 18],
            [19, 20, 23, 24],
            [21, 22, 25, 26],
        ])

        output = quadtree._concat(input)

        self.assertEqual(input.shape, (4, 2, 2))
        self.assertEqual(output.shape, (4, 4))
        np_test.assert_array_equal(output, target)

    def test_concat_dim3(self):
        octree = kdTree(spatial_dim=3)

        input = np.array([
            [[[1]]],
            [[[2]]],
            [[[3]]],
            [[[4]]],
            [[[5]]],
            [[[6]]],
            [[[7]]],
            [[[8]]],
        ])

        target = np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ])

        output = octree._concat(input)

        self.assertEqual(input.shape, (8, 1, 1, 1))
        self.assertEqual(output.shape, (2, 2, 2))
        np_test.assert_array_equal(output, target)

        input = np.array(
            [
                [[[11, 12], [13, 14]], [[15, 16], [17, 18]]],
                [[[19, 20], [21, 22]], [[23, 24], [25, 26]]],
                [[[27, 28], [29, 30]], [[31, 32], [33, 34]]],
                [[[35, 36], [37, 38]], [[39, 40], [41, 42]]],
                [[[43, 44], [45, 46]], [[47, 48], [49, 50]]],
                [[[51, 52], [53, 54]], [[55, 56], [57, 58]]],
                [[[59, 60], [61, 62]], [[63, 64], [65, 66]]],
                [[[67, 68], [69, 70]], [[71, 72], [73, 74]]],
            ]
        )

        target = np.array(
            [
                [[11, 12, 19, 20], [13, 14, 21, 22], [27, 28, 35, 36], [29, 30, 37, 38]],
                [[15, 16, 23, 24], [17, 18, 25, 26], [31, 32, 39, 40], [33, 34, 41, 42]],
                [[43, 44, 51, 52], [45, 46, 53, 54], [59, 60, 67, 68], [61, 62, 69, 70]],
                [[47, 48, 55, 56], [49, 50, 57, 58], [63, 64, 71, 72], [65, 66, 73, 74]]
            ],
        )

        output = octree._concat(input)

        self.assertEqual(input.shape, (8, 2, 2, 2))
        self.assertEqual(output.shape, (4, 4, 4))
        np_test.assert_array_equal(output, target)


class TestWhiteboxReversibleSplitConcat(unittest.TestCase):
    """ Tests the reversibility of the split and concat functions for quadtrees and octrees.

    Performs first a split and second a concatination of the input data.
    The resulting data should have exactly the same shape and contain the same values as the input.
    """
    def test_split_concat_dim2(self):
        quadtree = kdTree(spatial_dim=2)

        elements = np.array([
            [11, 12, 13, 14],
            [15, 16, 17, 18],
            [19, 20, 21, 22],
            [23, 24, 25, 26],
        ])

        output = quadtree._concat(quadtree._split(elements))

        self.assertEqual(elements.shape, (4, 4))
        self.assertEqual(output.shape, elements.shape)
        np_test.assert_array_equal(output, elements)

    def test_split_concat_dim3(self):
        octree = kdTree(spatial_dim=3)

        elements = np.array(
            [
                [[11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22], [23, 24, 25, 26]],
                [[27, 28, 29, 30], [31, 32, 33, 34], [35, 36, 37, 38], [39, 40, 41, 42]],
                [[43, 44, 45, 46], [47, 48, 49, 50], [51, 52, 53, 54], [55, 56, 57, 58]],
                [[59, 60, 61, 62], [63, 64, 65, 66], [67, 68, 69, 70], [71, 72, 73, 74]],
            ]
        )

        output = octree._concat(octree._split(elements))

        self.assertEqual(elements.shape, (4, 4, 4))
        self.assertEqual(output.shape, elements.shape)
        np_test.assert_array_equal(output, elements)


class TestWhiteboxReversibleConcatSplit(unittest.TestCase):
    """ Tests the reversibility of the concat and split functions for quadtrees and octrees.

    Performs first a concatination and second a split of the input data.
    The resulting data should have exactly the same shape and contain the same values as the input.
    """
    def test_concat_split_dim2(self):
        quadtree = kdTree(spatial_dim=2)

        elements = np.array([
            [[11, 12], [13, 14]],
            [[15, 16], [17, 18]],
            [[19, 20], [21, 22]],
            [[23, 24], [25, 26]],
        ])

        output = quadtree._split(quadtree._concat(elements))

        self.assertEqual(elements.shape, (4, 2, 2))
        self.assertEqual(output.shape, elements.shape)
        np_test.assert_array_equal(elements, output)

    def test_concat_split_dim3(self):
        octree = kdTree(spatial_dim=3)

        elements = np.array(
            [
                [[[11, 12], [13, 14]], [[15, 16], [17, 18]]],
                [[[19, 20], [21, 22]], [[23, 24], [25, 26]]],
                [[[27, 28], [29, 30]], [[31, 32], [33, 34]]],
                [[[35, 36], [37, 38]], [[39, 40], [41, 42]]],
                [[[43, 44], [45, 46]], [[47, 48], [49, 50]]],
                [[[51, 52], [53, 54]], [[55, 56], [57, 58]]],
                [[[59, 60], [61, 62]], [[63, 64], [65, 66]]],
                [[[67, 68], [69, 70]], [[71, 72], [73, 74]]],
            ]
        )

        output = octree._split(octree._concat(elements))

        self.assertEqual(elements.shape, (8, 2, 2, 2))
        self.assertEqual(output.shape, elements.shape)
        np_test.assert_array_equal(elements, output)

