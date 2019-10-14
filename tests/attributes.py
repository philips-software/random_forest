import unittest
from mpyc.runtime import mpc
from src.attributes import select_best_attribute, partition_on

s = mpc.SecInt()
output = mpc.output
run = mpc.run


class AttributeTests(unittest.TestCase):
    def test_select_best_attribute(self):
        samples = [
            [s(0), s(1), s(1), s(0)],
            [s(1), s(0), s(1), s(1)],
            [s(0), s(0), s(0), s(1)],
        ]
        outcomes = [s(1), s(1), s(0)]
        best_attribute = select_best_attribute(samples, outcomes)
        self.assertEqual(run(output(best_attribute)), 2)

    def test_select_best_attribute_no_samples(self):
        self.assertRaises(ValueError, select_best_attribute, [], [])

    def test_select_best_attribute_with_gini_denominator_zero(self):
        samples = [
            [s(0), s(0), s(1), s(0)],
            [s(0), s(0), s(1), s(0)],
            [s(0), s(0), s(0), s(0)],
        ]
        outcomes = [s(1), s(1), s(0)]
        best_attribute = select_best_attribute(samples, outcomes)
        self.assertEqual(run(output(best_attribute)), 2)

    def test_partition_on(self):
        samples = [
            [s(0), s(1), s(1), s(0)],
            [s(1), s(0), s(1), s(1)],
            [s(0), s(0), s(0), s(1)],
        ]
        best_attribute = s(2)
        best_attribute_value = s(0)
        is_active = [s(1), s(1), s(1)]
        left_active, right_active = partition_on(
            samples, is_active, best_attribute, best_attribute_value)
        self.assertEqual(run(output(left_active)), [0, 0, 1])
        self.assertEqual(run(output(right_active)), [1, 1, 0])
