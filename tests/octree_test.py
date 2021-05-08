import unittest
from utils.octree import Octree


class TestConversionTrinaryDecimal(unittest.TestCase):
    def test_dec_to_tri(self):
        repr = Octree()._dec_to_tri(1)
        self.assertEqual(repr, [1, 1, 1, 1, 1, 1, 1, 1])

        repr = Octree()._dec_to_tri(2)
        self.assertEqual(repr, [1, 1, 1, 1, 1, 1, 1, 2])

        repr = Octree()._dec_to_tri(3)
        self.assertEqual(repr, [1, 1, 1, 1, 1, 1, 1, 3])

        repr = Octree()._dec_to_tri(4)
        self.assertEqual(repr, [1, 1, 1, 1, 1, 1, 2, 1])

        repr = Octree()._dec_to_tri(5)
        self.assertEqual(repr, [1, 1, 1, 1, 1, 1, 2, 2])

        repr = Octree()._dec_to_tri(6)
        self.assertEqual(repr, [1, 1, 1, 1, 1, 1, 2, 3])

        repr = Octree()._dec_to_tri(7)
        self.assertEqual(repr, [1, 1, 1, 1, 1, 1, 3, 1])

        repr = Octree()._dec_to_tri(8)
        self.assertEqual(repr, [1, 1, 1, 1, 1, 1, 3, 2])

        repr = Octree()._dec_to_tri(9)
        self.assertEqual(repr, [1, 1, 1, 1, 1, 1, 3, 3])

        repr = Octree()._dec_to_tri(10)
        self.assertEqual(repr, [1, 1, 1, 1, 1, 2, 1, 1])

        repr = Octree()._dec_to_tri(100)
        self.assertEqual(repr, [1, 1, 1, 2, 1, 3, 1, 1])

        repr = Octree()._dec_to_tri(1000)
        self.assertEqual(repr, [1, 2, 2, 1, 2, 1, 1, 1])

        repr = Octree()._dec_to_tri(2853)
        self.assertEqual(repr, [2, 1, 3, 3, 1, 2, 3, 3])

        repr = Octree()._dec_to_tri(6561)
        self.assertEqual(repr, [3, 3, 3, 3, 3, 3, 3, 3])

    def test_tri_to_dec(self):
        repr = Octree()._tri_to_dec([1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(repr, 1)

        repr = Octree()._tri_to_dec([1, 1, 1, 1, 1, 1, 1, 2])
        self.assertEqual(repr, 2)

        repr = Octree()._tri_to_dec([1, 1, 1, 1, 1, 1, 1, 3])
        self.assertEqual(repr, 3)

        repr = Octree()._tri_to_dec([1, 1, 1, 1, 1, 1, 2, 1])
        self.assertEqual(repr, 4)

        repr = Octree()._tri_to_dec([1, 1, 1, 1, 1, 1, 2, 2])
        self.assertEqual(repr, 5)

        repr = Octree()._tri_to_dec([1, 1, 1, 1, 1, 1, 2, 3])
        self.assertEqual(repr, 6)

        repr = Octree()._tri_to_dec([1, 1, 1, 1, 1, 1, 3, 1])
        self.assertEqual(repr, 7)

        repr = Octree()._tri_to_dec([1, 1, 1, 1, 1, 1, 3, 2])
        self.assertEqual(repr, 8)

        repr = Octree()._tri_to_dec([1, 1, 1, 1, 1, 1, 3, 3])
        self.assertEqual(repr, 9)

        repr = Octree()._tri_to_dec([1, 1, 1, 1, 1, 2, 1, 1])
        self.assertEqual(repr, 10)

        repr = Octree()._tri_to_dec([1, 1, 1, 2, 1, 3, 1, 1])
        self.assertEqual(repr, 100)

        repr = Octree()._tri_to_dec([1, 2, 2, 1, 2, 1, 1, 1])
        self.assertEqual(repr, 1000)

        repr = Octree()._tri_to_dec([2, 1, 3, 3, 1, 2, 3, 3])
        self.assertEqual(repr, 2853)

        repr = Octree()._tri_to_dec([3, 3, 3, 3, 3, 3, 3, 3])
        self.assertEqual(repr, 6561)
