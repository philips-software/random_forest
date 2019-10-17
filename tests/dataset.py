import unittest
from src.dataset import ObliviousDataset, Sample
from mpyc.runtime import mpc
from src.output import output

s = mpc.SecInt()


def sample(*inputs):
    return Sample(inputs, 0)


class ObliviousDatasetTest(unittest.TestCase):
    def test_initialize_with_list(self):
        self.assertEqual(
            ObliviousDataset(sample(1, 2), sample(3, 4)),
            ObliviousDataset([sample(1, 2), sample(3, 4)])
        )
        self.assertEqual(
            ObliviousDataset(sample(1, 2)),
            ObliviousDataset([sample(1, 2)])
        )

    def test_len(self):
        self.assertEqual(len(ObliviousDataset()), 0)
        self.assertEqual(len(ObliviousDataset(
            [sample(0), sample(1), sample(2)])), 3)

    def test_row_indexing(self):
        data = ObliviousDataset(sample(0, 1), sample(2, 3))
        self.assertEqual(data[0], sample(0, 1))
        self.assertEqual(data[1], sample(2, 3))

    def test_column(self):
        dataset = ObliviousDataset(
            sample(s(0),  s(1),  s(2)),
            sample(s(10), s(11), s(12)),
            sample(s(20), s(21), s(22))
        )
        self.assertEqual(reveal(dataset.column(s(0))), [0, 10, 20])
        self.assertEqual(reveal(dataset.column(s(1))), [1, 11, 21])
        self.assertEqual(reveal(dataset.column(s(2))), [2, 12, 22])

    def test_reveal_all_rows(self):
        dataset = ObliviousDataset(
            sample(s(0),  s(1),  s(2)),
            sample(s(10), s(11), s(12)),
            sample(s(20), s(21), s(22))
        )
        self.assertEqual(
            reveal(dataset),
            [
                [0,  1,  2],
                [10, 11, 12],
                [20, 21, 22]
            ]
        )

    def test_empty_subset(self):
        dataset = ObliviousDataset(
            sample(s(0),  s(1),  s(2)),
            sample(s(10), s(11), s(12)),
            sample(s(20), s(21), s(22))
        ).subset([s(0), s(0), s(0)])
        self.assertEqual(
            reveal(dataset),
            []
        )

    def test_subset(self):
        dataset = ObliviousDataset(
            sample(s(0),  s(1),  s(2)),
            sample(s(10), s(11), s(12)),
            sample(s(20), s(21), s(22))
        ).subset([s(1), s(0), s(1)])
        self.assertEqual(
            reveal(dataset),
            [
                [0,  1,  2],
                [20, 21, 22]
            ]
        )

    def test_subset_of_subset(self):
        dataset = ObliviousDataset(
            sample(s(0),  s(1),  s(2)),
            sample(s(10), s(11), s(12)),
            sample(s(20), s(21), s(22))
        ) \
            .subset([s(1), s(0), s(1)]) \
            .subset([s(0), s(1), s(1)])
        self.assertEqual(
            reveal(dataset),
            [
                [20, 21, 22]
            ]
        )

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
        ).subset([s(1), s(0), s(1)])
        self.assertEqual(reveal(dataset.is_active(0)), True)
        self.assertEqual(reveal(dataset.is_active(1)), False)


def reveal(secret):
    return mpc.run(output(secret))
