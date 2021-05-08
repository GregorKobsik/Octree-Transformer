import unittest
import numpy as np
import numpy.testing as np_test
import matplotlib.image as mpimg
from utils.quadtree import Quadtree


def binarize(x):
    return np.add(x > 0.1, x == 1, dtype=int) + 1


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
        qtree = Quadtree().insert_image(mnist_28x28())
        np_test.assert_array_equal(qtree.north_west.north_west.value, 1)
        np_test.assert_array_equal(qtree.south_west.south_west.value, 1)

    def test_mnist_32x32(self):
        qtree = Quadtree().insert_image(mnist_32x32())
        np_test.assert_array_equal(qtree.north_west.north_west.value, 1)
        np_test.assert_array_equal(qtree.south_west.south_west.value, 1)
        np_test.assert_array_almost_equal(qtree.south_west.north_east.north_east.north_east.value, 3)


class TestImageRetrival(unittest.TestCase):
    def test_image_shape_28x28(self):
        img = mnist_28x28()
        qtree = Quadtree().insert_image(img)
        self.assertEqual(qtree.get_image(10).shape, img.shape)

    def test_image_shape_32x32(self):
        img = mnist_32x32()
        qtree = Quadtree().insert_image(img)
        self.assertEqual(qtree.get_image(10).shape, img.shape)

    def test_image_shape_32x32_binarized(self):
        img_bin = mnist_32x32_binarized()
        qtree = Quadtree().insert_image(img_bin)
        self.assertEqual(qtree.get_image(10).shape, img_bin.shape)

    def test_image2sequence2image_shape(self):
        img_bin = mnist_32x32_binarized()
        qtree = Quadtree().insert_image(img_bin)
        qtree = Quadtree().insert_sequence(qtree.get_sequence(), resolution=img_bin.shape)
        np_test.assert_array_equal(qtree.get_image(10).shape, img_bin.shape)


class TestSequenceRetrival(unittest.TestCase):
    def test_sequence_depth3(self):
        seq = "222221112112112122121"
        qtree = Quadtree().insert_image(mnist_32x32_binarized())
        self.assertSequenceEqual(''.join(str(x) for x in qtree.get_sequence(depth=3)), seq)

    def test_diagonal_sequence(self):
        seq = "2" + "1221" + "12211221" + "1221122112211221" + "12211221122112211221122112211221" \
            + "1221122112211221122112211221122112211221122112211221122112211221"
        shape = (32, 32)
        qtree = Quadtree().insert_sequence(seq, shape)
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
        seq = [2, 1, 1, 1, 1, 1]
        shape = (32, 32)
        qtree = Quadtree().insert_sequence(seq, shape, autorepair_errors=True, silent=True)
        qtree.get_image(1)
        self.assertTrue(True)

    def test_error_input_array_dimensions_for_the_concatenation_axis_must_match_exactly_1(self):
        """ Test should not throw any error and the misshaped input should be automatically repaired.

        Error resolved, when final nodes are assigned at resolution (1,1), too.
        """
        seq = [
            2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1,
            2, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2,
            2, 2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1,
            2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1,
            2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1
        ]
        shape = (32, 32)
        qtree = Quadtree().insert_sequence(seq, shape, autorepair_errors=True, silent=True)
        qtree.get_image(1)
        self.assertTrue(True)
