import unittest
from mpyc.runtime import mpc
from src.maximum import index_of_maximum

s = mpc.SecInt()
output = mpc.output
run = mpc.run


class MaximumTest(unittest.TestCase):

    def test_maximum_with_two_arguments(self):
        self.assertEqual(index_of_maximum((1, 2), (1, 3)), 0)
        self.assertEqual(index_of_maximum((1, 4), (1, 3)), 1)

    def test_maximum_with_more_arguments(self):
        self.assertEqual(index_of_maximum((1, 4), (1, 3), (1, 2)), 2)

    def test_maximum_with_single_argument(self):
        self.assertEqual(index_of_maximum((1, 1)), 0)

    def test_maximum_with_no_arguments(self):
        self.assertRaises(ValueError, index_of_maximum)

    def test_maximum_mpc(self):
        index = index_of_maximum((s(1), s(4)), (s(1), s(3)), (s(1), s(2)))
        self.assertEqual(run(output(index)), 2)
