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

    def test_random_sample(self):
        dataset = ObliviousDataset.create(
            Sample([s(1), s(2), s(3)], s(4))
        )
        self.assertEqual(reveal(dataset.random_sample()), Sample([1, 2, 3], 4))

    def test_random_sample(self):
        dataset = ObliviousDataset.create(
            Sample([s(1), s(2), s(3)], s(4)),
            Sample([s(11), s(12), s(13)], s(14))
        )
        seenFirst = False
        seenSecond = False
        for i in range(10):
            seenFirst = seenFirst or reveal(dataset.random_sample()
                                            ) == Sample([1, 2, 3], 4)
            seenSecond = seenSecond or reveal(dataset.random_sample()
                                              ) == Sample([11, 12, 13], 14)
        self.assertTrue(seenFirst)
        self.assertTrue(seenSecond)
