import unittest

from src.maximum import maximum
from src.secint import secint as s
from tests.reveal import reveal


def q(numerator, denominator):
    return s(numerator), s(denominator)


class MaximumTest(unittest.TestCase):

    def test_maximum_with_two_arguments(self):
        self.assertEqual(
            reveal(maximum(q(1, 2), q(1, 3))), ((1, 2), 0))
        self.assertEqual(
            reveal(maximum(q(1, 4), q(1, 3))), ((1, 3), 1))

    def test_maximum_with_more_arguments(self):
        self.assertEqual(reveal(maximum(
            q(1, 4), q(1, 3), q(1, 2))), ((1, 2), 2))

    def test_maximum_with_single_argument(self):
        self.assertEqual(reveal(maximum(q(1, 1))), ((1, 1), 0))

    def test_maximum_with_no_arguments(self):
        self.assertRaises(ValueError, maximum)
