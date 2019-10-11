import unittest
from mpyc.runtime import mpc
from src.attributes import select_best_attribute, partition_on

s = mpc.SecInt()
output = mpc.output
run = mpc.run


class AttributeTests(unittest.TestCase):
    def test_select_best_attribute(self):
        samples = [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [0, 0, 0, 1],
        ]
        outcomes = [1, 1, 0]
        self.assertEqual(select_best_attribute(samples, outcomes), 2)

    def test_select_best_attribute_no_samples(self):
        self.assertRaises(ValueError, select_best_attribute, [], [])

    def test_select_best_attribute_with_gini_denominator_zero(self):
        samples = [
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ]
        outcomes = [1, 1, 0]
        self.assertEqual(select_best_attribute(samples, outcomes), 2)

    def test_select_best_attribute_mpc(self):
        samples = [
            [s(0), s(1), s(1), s(0)],
            [s(1), s(0), s(1), s(1)],
            [s(0), s(0), s(0), s(1)],
        ]
        outcomes = [s(1), s(1), s(0)]
        best_attribute = select_best_attribute(samples, outcomes)
        self.assertEqual(run(output(best_attribute)), 2)

    def test_partition_on(self):
        samples = [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [0, 0, 0, 1],
        ]
        best_attribute = 2
        best_attribute_value = 0
        left, right = partition_on(
            samples, best_attribute, best_attribute_value)
        self.assertEqual(left, [samples[2]])
        self.assertEqual(right, [samples[0], samples[1]])
