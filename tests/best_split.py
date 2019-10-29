import unittest
from mpyc.runtime import mpc
from src.dataset import ObliviousDataset, Sample
from src.best_split import select_best_attribute

s = mpc.SecInt()
output = mpc.output
run = mpc.run


class AttributeTests(unittest.TestCase):
    def test_select_best_attribute(self):
        samples = ObliviousDataset(
            Sample([s(0), s(1), s(1), s(0)], s(1)),
            Sample([s(1), s(0), s(1), s(1)], s(1)),
            Sample([s(0), s(0), s(0), s(1)], s(0))
        )
        best_attribute = select_best_attribute(samples)
        self.assertEqual(run(output(best_attribute)), 2)

    def test_select_best_attribute_with_gini_denominator_zero(self):
        samples = ObliviousDataset(
            Sample([s(0), s(0), s(1), s(0)], s(1)),
            Sample([s(0), s(0), s(1), s(0)], s(1)),
            Sample([s(0), s(0), s(0), s(0)], s(0))
        )
        best_attribute = select_best_attribute(samples)
        self.assertEqual(run(output(best_attribute)), 2)

    def test_select_best_attribute_using_subset(self):
        samples = ObliviousDataset(
            Sample([s(0), s(1), s(1), s(0)], s(1)),
            Sample([s(1), s(0), s(1), s(1)], s(1)),
            Sample([s(42), s(43), s(44), s(45)], s(46)),
            Sample([s(0), s(0), s(0), s(1)], s(0)),
        ).select([s(0), s(1), s(0), s(1)])
        best_attribute = select_best_attribute(samples)
        self.assertEqual(run(output(best_attribute)), 2)
