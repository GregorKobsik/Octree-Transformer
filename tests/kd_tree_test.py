import unittest
import numpy as np
import numpy.testing as np_test
import matplotlib.image as mpimg

from utils import kdTree


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


class TestQuadtree(unittest.TestCase):
    """ Testsuit for a 2d-tree, aka a quadtree. """
    def binarize(self, x):
        """ Binarize the input array 'x'.

        For each pixel assign:
        '0' if the color value is below 0.1, else '1'
        """
        return np.array(x > 0.1, dtype=int)

    def mnist_28x28(self):
        """ Load a single MNIST image.

        The color values are scaled to the range [0, 1] with only one color channel.
        """
        return mpimg.imread('tests/img/mnist.jpg')[:, :, 0] / 255.0

    def mnist_28x28_binarized(self):
        """ Return a single binarized MNIST image, with a resolution of 28x28. """
        return self.binarize(self.mnist_28x28())

    def mnist_32x32(self):
        """ Return a single padded MNIST image, with a resolution of 32x32. """
        return np.pad(self.mnist_28x28(), (2, ))

    def mnist_32x32_binarized(self):
        """ Return a single binarized and padded MNIST image, with a resolution of 32x32. """
        return self.binarize(self.mnist_32x32())

    def test_data_available(self):
        """ Check if mnist image is available. """
        self.assertEqual(self.mnist_28x28().shape, (28, 28))

    def test_insert_image_28x28(self):
        """ Create a quadtree and try to insert image data.

        Raw MNIST images have a resolution of 28 x 28.
        The data is only divisible by 2 up to the second depth layer.
        Anything above this, should result in an error.
        """
        qtree = kdTree(spatial_dim=2).insert_element_array(self.mnist_28x28(), max_depth=1)

        self.assertEqual(qtree.child_nodes[0].child_nodes[0].value, 1)
        self.assertEqual(qtree.child_nodes[2].child_nodes[2].value, 1)

    def test_insert_image_32x32(self):
        """ Create a quadtree and try to insert image data.

        The MNIST image is padded on every sidy by 2 empty pixels,
        thus it should create a valid quadtree up to a depth of 5.
        The code should not throw any errors, if requests to create higher depth quadtrees.
        It should be able to detect the maximum depth and return a valid quadtree.
        """
        qtree = kdTree(spatial_dim=2).insert_element_array(self.mnist_32x32())

        self.assertEqual(qtree.child_nodes[0].child_nodes[0].value, 1)
        self.assertEqual(qtree.child_nodes[2].child_nodes[2].value, 1)
        self.assertEqual(qtree.child_nodes[2].child_nodes[1].child_nodes[1].child_nodes[1].value, 3)

    def test_image_retrival(self):
        """ Inserts and retrives a binarized image array. The output should be identical to the input. """
        input = self.mnist_32x32_binarized()
        qtree = kdTree(spatial_dim=2).insert_element_array(input)
        output = qtree.get_element_array()

        self.assertEqual(input.shape, output.shape)
        np_test.assert_array_equal(input, output)

    def test_image_to_sequence_to_image(self):
        """ Transforms an image array into a token sequence and back to an image array.

        The resulting data should not change in shape and values.
        """
        input = self.mnist_32x32_binarized()

        qtree = kdTree(spatial_dim=2).insert_element_array(input)
        token_sequence = qtree.get_token_sequence()[0]

        qtree2 = kdTree(spatial_dim=2).insert_token_sequence(token_sequence, resolution=input.shape[0])
        output = qtree2.get_element_array()

        np_test.assert_array_equal(input, output)

    def test_token_sequence_generation_depth2(self):
        """ Inputs an binarized image and generates a token sequence up to a depth of 2. """
        input = self.mnist_32x32_binarized()
        target = "2222" + "1112112112122121"

        qtree = kdTree(spatial_dim=2).insert_element_array(input)
        output = qtree.get_token_sequence(depth=2)[0]

        self.assertSequenceEqual(target, ''.join(str(x) for x in output))

    def test_token_sequence_retrival_short(self):
        """ Inserts a sequence representing a diagonal line and tries to retrive the same token sequence. """
        input = ("1221" + "12211221")

        qtree = kdTree(spatial_dim=2).insert_token_sequence(input, resolution=32)
        output = qtree.get_token_sequence()[0]

        self.assertEqual(len(input), len(output))
        self.assertSequenceEqual(input, ''.join(str(x) for x in output))

    def test_token_sequence_retrival_diagonal(self):
        """ Inserts a sequence representing a diagonal line and tries to retrive the same token sequence. """
        input = (
            "1221" + "12211221" + "1221122112211221" + "12211221122112211221122112211221" +
            "1221122112211221122112211221122112211221122112211221122112211221"
        )

        qtree = kdTree(spatial_dim=2).insert_token_sequence(input, resolution=32)
        output = qtree.get_token_sequence()[0]

        self.assertEqual(len(input), len(output))
        self.assertSequenceEqual(input, ''.join(str(x) for x in output))

    def test_autorepair_error_pop_from_empty_list(self):
        """ Test should not throw any error and the misshaped input should be automatically repaired.

        Implementation error - whitebox test.
        Error resolved: Check the size of the 'open_set' to decide if the parser is done.
        """
        input = [1, 1, 1, 1, 1]

        qtree = kdTree(spatial_dim=2).insert_token_sequence(input, resolution=32, autorepair_errors=True, silent=True)
        qtree.get_element_array(depth=1)

        self.assertTrue(True)

    def test_autorepair_error_input_array_dimensions_for_the_concatenation_axis_must_match_exactly(self):
        """ Test should not throw any error and the misshaped input should be automatically repaired.

        Implementation error - whitebox test.
        Error resolved: Nodes should be assigned final at a resolution of (1,1), too.
        """
        input = [
            1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1, 2,
            1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2,
            2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2,
            1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2,
            1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1
        ]

        qtree = kdTree(spatial_dim=2).insert_token_sequence(input, resolution=32, autorepair_errors=True, silent=True)
        qtree.get_element_array(depth=1)

        self.assertTrue(True)

    def test_token_sequence_retrival_value_depth(self):
        """ Inputs an binarized image and generates a value and depth token sequence. """
        input = self.mnist_32x32_binarized()

        target_value = [2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1]
        target_depth = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

        qtree = kdTree(spatial_dim=2).insert_element_array(input)
        output_value, output_depth = qtree.get_token_sequence(depth=2, return_depth=True)

        np_test.assert_array_equal(target_value, output_value)
        np_test.assert_array_equal(target_depth, output_depth)

    def test_token_sequence_retrival_value_pos(self):
        """ Inputs an binarized image and generates a value and position token sequence. """
        input = self.mnist_32x32_binarized()

        target_value = [2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1]
        target_pos = [
            [16, 16], [16, 48], [48, 16], [48, 48], [8, 8], [8, 24], [24, 8], [24, 24], [8, 40], [8, 56], [24, 40],
            [24, 56], [40, 8], [40, 24], [56, 8], [56, 24], [40, 40], [40, 56], [56, 40], [56, 56]
        ]

        qtree = kdTree(spatial_dim=2).insert_element_array(input)
        output_value, output_pos = qtree.get_token_sequence(depth=2, return_pos=True)

        np_test.assert_array_equal(target_value, output_value)
        np_test.assert_array_equal(target_pos, output_pos)

    def test_token_sequence_retrival_value_depth_pos(self):
        """ Inputs an binarized image and generates a value, depth and position token sequence. """
        input = self.mnist_32x32_binarized()

        target_value = [2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1]
        target_depth = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        target_pos = [
            [16, 16], [16, 48], [48, 16], [48, 48], [8, 8], [8, 24], [24, 8], [24, 24], [8, 40], [8, 56], [24, 40],
            [24, 56], [40, 8], [40, 24], [56, 8], [56, 24], [40, 40], [40, 56], [56, 40], [56, 56]
        ]

        qtree = kdTree(spatial_dim=2).insert_element_array(input)
        output_value, output_depth, output_pos = qtree.get_token_sequence(depth=2, return_depth=True, return_pos=True)

        np_test.assert_array_equal(target_value, output_value)
        np_test.assert_array_equal(target_depth, output_depth)
        np_test.assert_array_equal(target_pos, output_pos)
