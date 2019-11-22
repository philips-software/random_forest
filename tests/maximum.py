import unittest

from src.maximum import index_of_maximum
from src.secint import secint as s
from tests.reveal import reveal


class MaximumTest(unittest.TestCase):

    def test_maximum_with_two_arguments(self):
        self.assertEqual(
            reveal(index_of_maximum((s(1), s(2)), (s(1), s(3)))), 0)
        self.assertEqual(
            reveal(index_of_maximum((s(1), s(4)), (s(1), s(3)))), 1)

    def test_maximum_with_more_arguments(self):
        self.assertEqual(reveal(index_of_maximum(
            (s(1), s(4)), (s(1), s(3)), (s(1), s(2)))), 2)

    def test_maximum_with_single_argument(self):
        self.assertEqual(reveal(index_of_maximum((s(1), s(1)))), 0)

    def test_maximum_with_no_arguments(self):
        self.assertRaises(ValueError, index_of_maximum)
