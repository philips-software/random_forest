import unittest
from mpyc.runtime import mpc
from src.best_split import select_best_attribute

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
