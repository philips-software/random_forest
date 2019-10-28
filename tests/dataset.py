import unittest
from src.dataset import ObliviousDataset, Sample
from mpyc.runtime import mpc
from src.output import output

s = mpc.SecInt()


def sample(*inputs):
    return Sample(inputs, s(0))


class ObliviousDatasetTest(unittest.TestCase):
    def test_len(self):
        self.assertEqual(len(ObliviousDataset()), 0)
        self.assertEqual(len(ObliviousDataset(
            [sample(0), sample(1), sample(2)])), 3)

    def test_row_indexing(self):
        data = ObliviousDataset(sample(s(0), s(1)), sample(s(2), s(3)))
        self.assertEqual(reveal(data[0]), Sample([0, 1], 0))
        self.assertEqual(reveal(data[1]), Sample([2, 3], 0))

    def test_column_with_public_index(self):
        dataset = ObliviousDataset(
            sample(s(0),  s(1),  s(2)),
            sample(s(10), s(11), s(12)),
            sample(s(20), s(21), s(22))
        )
        self.assertEqual(reveal(dataset.column(0)), [0, 10, 20])
        self.assertEqual(reveal(dataset.column(1)), [1, 11, 21])
        self.assertEqual(reveal(dataset.column(2)), [2, 12, 22])

    def test_column_of_subset_with_public_index(self):
        dataset = ObliviousDataset(
            sample(s(0),  s(1),  s(2)),
            sample(s(10), s(11), s(12)),
            sample(s(20), s(21), s(22))
        ).select([s(1), s(0), s(1)])
        self.assertEqual(reveal(dataset.column(1)), [1, 21])

    def test_column_with_secret_index(self):
        dataset = ObliviousDataset(
            sample(s(0),  s(1),  s(2)),
            sample(s(10), s(11), s(12)),
            sample(s(20), s(21), s(22))
        )
        self.assertEqual(reveal(dataset.column(s(0))), [0, 10, 20])
        self.assertEqual(reveal(dataset.column(s(1))), [1, 11, 21])
        self.assertEqual(reveal(dataset.column(s(2))), [2, 12, 22])

    def test_column_of_subset_with_secret_index(self):
        dataset = ObliviousDataset(
            sample(s(0),  s(1),  s(2)),
            sample(s(10), s(11), s(12)),
            sample(s(20), s(21), s(22))
        ).select([s(1), s(0), s(1)])
        self.assertEqual(reveal(dataset.column(s(1))), [1, 21])

    def test_outcomes(self):
        dataset = ObliviousDataset(
            Sample([s(0),  s(1),  s(2)], outcome=s(60)),
            Sample([s(10), s(11), s(12)], outcome=s(70)),
            Sample([s(20), s(21), s(22)], outcome=s(80))
        ).select([s(1), s(0), s(1)])
        self.assertEqual(reveal(dataset.outcomes), [60, 80])

    def test_is_active(self):
        dataset = ObliviousDataset(
            sample(s(0),  s(1),  s(2))
        )
        self.assertEqual(reveal(dataset.is_active(0)), True)

    def test_is_active_with_subset(self):
        dataset = ObliviousDataset(
            sample(s(0),  s(1),  s(2)),
            sample(s(10), s(11), s(12)),
            sample(s(20), s(21), s(22))
        ).select([s(1), s(0), s(1)])
        self.assertEqual(reveal(dataset.is_active(0)), True)
        self.assertEqual(reveal(dataset.is_active(1)), False)


def reveal(secret):
    return mpc.run(output(secret))
