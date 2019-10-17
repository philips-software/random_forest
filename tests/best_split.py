import unittest
from mpyc.runtime import mpc
from src.dataset import ObliviousDataset
from src.best_split import select_best_attribute

s = mpc.SecInt()
output = mpc.output
run = mpc.run


class AttributeTests(unittest.TestCase):
    def test_select_best_attribute(self):
        samples = ObliviousDataset(
            [s(0), s(1), s(1), s(0)],
            [s(1), s(0), s(1), s(1)],
            [s(0), s(0), s(0), s(1)],
        )
        outcomes = [s(1), s(1), s(0)]
        best_attribute = select_best_attribute(samples, outcomes)
        self.assertEqual(run(output(best_attribute)), 2)

    def test_select_best_attribute_no_samples(self):
        samples = ObliviousDataset()
        self.assertRaises(ValueError, select_best_attribute, samples, [])

    def test_select_best_attribute_with_gini_denominator_zero(self):
        samples = ObliviousDataset(
            [s(0), s(0), s(1), s(0)],
            [s(0), s(0), s(1), s(0)],
            [s(0), s(0), s(0), s(0)],
        )
        outcomes = [s(1), s(1), s(0)]
        best_attribute = select_best_attribute(samples, outcomes)
        self.assertEqual(run(output(best_attribute)), 2)

    def test_select_best_attribute_using_subset(self):
        samples = ObliviousDataset(
            [s(0), s(1), s(1), s(0)],
            [s(1), s(0), s(1), s(1)],
            [s(42), s(43), s(44), s(45)],
            [s(0), s(0), s(0), s(1)],
        ).subset([s(0), s(1), s(0), s(1)])
        outcomes = [s(1), s(1), s(46), s(0)]
        best_attribute = select_best_attribute(samples, outcomes)
        self.assertEqual(run(output(best_attribute)), 2)
