import unittest

from src.best_split import select_best_attribute
from src.dataset import ObliviousDataset, Sample
from src.secint import secint as s
from tests.reveal import reveal


class AttributeTests(unittest.TestCase):
    def test_select_best_attribute(self):
        samples = ObliviousDataset.create(
            Sample([s(0), s(1), s(1), s(0)], s(1)),
            Sample([s(1), s(0), s(1), s(1)], s(1)),
            Sample([s(0), s(0), s(0), s(1)], s(0))
        )
        (best_attribute, threshold) = select_best_attribute(samples)
        self.assertEqual(reveal(best_attribute), 2)
        self.assertEqual(reveal(threshold), 0)

    def test_select_best_attribute_with_gini_denominator_zero(self):
        samples = ObliviousDataset.create(
            Sample([s(0), s(0), s(1), s(0)], s(1)),
            Sample([s(0), s(0), s(1), s(0)], s(1)),
            Sample([s(0), s(0), s(0), s(0)], s(0))
        )
        (best_attribute, _) = select_best_attribute(samples)
        self.assertEqual(reveal(best_attribute), 2)

    def test_select_best_attribute_using_subset(self):
        samples = ObliviousDataset.create(
            Sample([s(0), s(1), s(1), s(0)], s(1)),
            Sample([s(1), s(0), s(1), s(1)], s(1)),
            Sample([s(42), s(43), s(44), s(45)], s(46)),
            Sample([s(0), s(0), s(0), s(1)], s(0)),
        ).select([s(0), s(1), s(0), s(1)])
        (best_attribute, _) = select_best_attribute(samples)
        self.assertEqual(reveal(best_attribute), 2)

    def test_select_best_continuous_attribute(self):
        samples = ObliviousDataset.create(
            Sample([s(1)], s(0)),
            Sample([s(2)], s(0)),
            Sample([s(3)], s(0)),
            Sample([s(4)], s(1)),
            Sample([s(5)], s(1)),
            continuous=[True]
        )
        (best_attribute, threshold) = select_best_attribute(samples)
        self.assertEqual(reveal(best_attribute), 0)
        self.assertEqual(reveal(threshold), 3)

    def test_select_best_attribute_from_continuous_and_binary(self):
        samples = ObliviousDataset.create(
            Sample([s(0), s(1)], s(0)),
            Sample([s(0), s(2)], s(0)),
            Sample([s(1), s(3)], s(0)),
            Sample([s(1), s(4)], s(1)),
            Sample([s(1), s(5)], s(1)),
            continuous=[False, True]
        )
        (best_attribute, threshold) = select_best_attribute(samples)
        self.assertEqual(reveal(best_attribute), 1)
        self.assertEqual(reveal(threshold), 3)
