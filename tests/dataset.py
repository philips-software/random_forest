import unittest

from mpyc.runtime import mpc

from src.dataset import ObliviousDataset, Sample
from src.secint import secint as s
from tests.reveal import reveal


def sample(*inputs):
    return Sample(inputs, s(0))


class ObliviousDatasetTest(unittest.TestCase):
    def test_column_with_public_index(self):
        dataset = ObliviousDataset.create(
            sample(s(0),  s(1),  s(2)),
            sample(s(10), s(11), s(12)),
            sample(s(20), s(21), s(22))
        )
        self.assertEqual(reveal(dataset.column(0)), [0, 10, 20])
        self.assertEqual(reveal(dataset.column(1)), [1, 11, 21])
        self.assertEqual(reveal(dataset.column(2)), [2, 12, 22])

    def test_column_of_subset_with_public_index(self):
        dataset = ObliviousDataset.create(
            sample(s(0),  s(1),  s(2)),
            sample(s(10), s(11), s(12)),
            sample(s(20), s(21), s(22))
        ).select([s(1), s(0), s(1)])
        self.assertEqual(reveal(dataset.column(1)), [1, 21])

    def test_column_with_secret_index(self):
        dataset = ObliviousDataset.create(
            sample(s(0),  s(1),  s(2)),
            sample(s(10), s(11), s(12)),
            sample(s(20), s(21), s(22))
        )
        self.assertEqual(reveal(dataset.column(s(0))), [0, 10, 20])
        self.assertEqual(reveal(dataset.column(s(1))), [1, 11, 21])
        self.assertEqual(reveal(dataset.column(s(2))), [2, 12, 22])

    def test_column_of_subset_with_secret_index(self):
        dataset = ObliviousDataset.create(
            sample(s(0),  s(1),  s(2)),
            sample(s(10), s(11), s(12)),
            sample(s(20), s(21), s(22))
        ).select([s(1), s(0), s(1)])
        self.assertEqual(reveal(dataset.column(s(1))), [1, 21])

    def test_outcomes(self):
        dataset = ObliviousDataset.create(
            Sample([s(0),  s(1),  s(2)], outcome=s(60)),
            Sample([s(10), s(11), s(12)], outcome=s(70)),
            Sample([s(20), s(21), s(22)], outcome=s(80))
        ).select([s(1), s(0), s(1)])
        self.assertEqual(reveal(dataset.outcomes), [60, 80])

    def test_number_of_attributes(self):
        dataset = ObliviousDataset.create(
            sample(s(1), s(2), s(3)),
            sample(s(4), s(5), s(6))
        )
        self.assertEqual(dataset.number_of_attributes, 3)

    def test_number_of_attributes_empty_set(self):
        dataset = ObliviousDataset.create()
        self.assertEqual(dataset.number_of_attributes, 0)

    def test_random_sample_with_one_sample(self):
        dataset = ObliviousDataset.create(Sample([s(1), s(2), s(3)], s(4)))
        self.assertEqual(reveal(dataset.choice()),
                         Sample([1, 2, 3], 4))

    def test_random_sample(self):
        dataset = ObliviousDataset.create(
            Sample([s(1), s(2), s(3)], s(4)),
            Sample([s(11), s(12), s(13)], s(14))
        )
        randomSamples = [reveal(dataset.choice()) for _ in range(10)]
        self.assertIn(Sample([1, 2, 3], 4), randomSamples)
        self.assertIn(Sample([11, 12, 13], 14), randomSamples)

    def test_determine_class_single_sample(self):
        dataset = ObliviousDataset.create(Sample([s(0)], s(0)))
        self.assertEqual(reveal(dataset.determine_class()), 0)

    def test_determine_class_multiple_samples(self):
        dataset = ObliviousDataset.create(
            Sample([s(0)], s(0)),
            Sample([s(0)], s(1)),
            Sample([s(0)], s(1)))
        self.assertEqual(reveal(dataset.determine_class()), 1)


class SampleTest(unittest.TestCase):
    def test_add_samples(self):
        sample1 = Sample([s(1), s(2), s(3)], s(4))
        sample2 = Sample([s(5), s(6), s(7)], s(8))
        self.assertEqual(reveal(sample1 + sample2), Sample([6, 8, 10], 12))

    def test_multiply_samples(self):
        sample = Sample([s(1), s(2), s(3)], s(4))
        self.assertEqual(reveal(sample * s(2)), Sample([2, 4, 6], 8))
