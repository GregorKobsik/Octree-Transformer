import unittest
import numpy as np
import numpy.testing as np_test
import matplotlib.image as mpimg
from utils.quadtree import QuadTree


def binarize(x):
    return np.add(x > 0.1, x == 1, dtype=int)


def mnist_28x28():
    return mpimg.imread('tests/img/mnist.jpg')[:, :, 0] / 255.0


def mnist_28x28_binarized():
    return binarize(mnist_28x28())


def mnist_32x32():
    return np.pad(mnist_28x28(), (2, ))


def mnist_32x32_binarized():
    return binarize(mnist_32x32())


class TestDataAvailable(unittest.TestCase):
    def test_mnist_available(self):
        img = mnist_28x28()
        self.assertEqual(img.shape, (28, 28))


class TestQuadtreeCreation(unittest.TestCase):
    def test_mnist_28x28(self):
        qtree = QuadTree().insert_image(mnist_28x28())
        np_test.assert_array_equal(qtree.north_west.north_west.value, 0)
        np_test.assert_array_equal(qtree.south_west.south_west.value, 0)

    def test_mnist_32x32(self):
        qtree = QuadTree().insert_image(mnist_32x32())
        np_test.assert_array_equal(qtree.north_west.north_west.value, 0)
        np_test.assert_array_equal(qtree.south_west.south_west.value, 0)
        np_test.assert_array_almost_equal(qtree.south_west.north_east.north_east.north_east.value, 2)


class TestImageRetrival(unittest.TestCase):
    def test_image_shape_28x28(self):
        img = mnist_28x28()
        qtree = QuadTree().insert_image(img)
        self.assertEqual(qtree.get_image(10).shape, img.shape)

    def test_image_shape_32x32(self):
        img = mnist_32x32()
        qtree = QuadTree().insert_image(img)
        self.assertEqual(qtree.get_image(10).shape, img.shape)

    def test_image_shape_32x32_binarized(self):
        img_bin = mnist_32x32_binarized()
        qtree = QuadTree().insert_image(img_bin)
        self.assertEqual(qtree.get_image(10).shape, img_bin.shape)

    def test_image2sequence2image_shape(self):
        img_bin = mnist_32x32_binarized()
        qtree = QuadTree().insert_image(img_bin)
        qtree = QuadTree().insert_sequence(qtree.get_sequence(), resolution=img_bin.shape)
        np_test.assert_array_equal(qtree.get_image(10).shape, img_bin.shape)


class TestSequenceRetrival(unittest.TestCase):
    def test_sequence_depth2(self):
        seq = "111110001001001011010"
        qtree = QuadTree().insert_image(mnist_32x32_binarized())
        self.assertSequenceEqual(qtree.get_sequence(depth=2, as_string=True), seq)
        self.assertSequenceEqual(''.join(str(x) for x in qtree.get_sequence(depth=2)), seq)

    def test_diagonal_sequence(self):
        seq = "1" + "0110" + "01100110" + "0110011001100110" + "01100110011001100110011001100110" \
            + "0110011001100110011001100110011001100110011001100110011001100110"
        shape = (32, 32)
        qtree = QuadTree().insert_sequence(seq, shape)
        self.assertSequenceEqual(qtree.get_sequence(as_string=True), seq)
        self.assertSequenceEqual(''.join(str(x) for x in qtree.get_sequence()), seq)


class TestErroneousSequences(unittest.TestCase):
    """ Defines different input sequences, which cause errors when autoreparining misshaped input sequences.

    The Quadtree class should catch those errors and handle the correction without throwing errors.
    As the Quadtree class will be expanded to handle different input sequences, those basic inputs should
    always pass.
    """
    def test_error_pop_from_empty_list_1(self):
        """ Test should not throw any error and the misshaped input should be automatically repaired.

        Error resolved, when checking the size of the open_set to decide if the parser is done.
        """
        seq = [1, 0, 0, 0, 0, 0]
        shape = (32, 32)
        qtree = QuadTree().insert_sequence(seq, shape, autorepair_errors=True, silent=True)
        qtree.get_image(1)
        self.assertTrue(True)

    def test_error_input_array_dimensions_for_the_concatenation_axis_must_match_exactly_1(self):
        """ Test should not throw any error and the misshaped input should be automatically repaired.

        Error resolved, when final nodes are assigned at resolution (1,1), too.
        """
        seq = [
            1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0,
            1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1,
            1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0,
            1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0
        ]
        shape = (32, 32)
        qtree = QuadTree().insert_sequence(seq, shape, autorepair_errors=True, silent=True)
        qtree.get_image(1)
        self.assertTrue(True)
