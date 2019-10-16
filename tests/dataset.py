import unittest
from src.dataset import ObliviousDataset
from mpyc.runtime import mpc
from src.output import output

s = mpc.SecInt()


class ObliviousDatasetTest(unittest.TestCase):
    def test_initialize_with_list(self):
        self.assertEqual(
            ObliviousDataset([1, 2], [3, 4]),
            ObliviousDataset([[1, 2], [3, 4]])
        )

    def test_column(self):
        dataset = ObliviousDataset(
            [s(0),  s(1),  s(2)],
            [s(10), s(11), s(12)],
            [s(20), s(21), s(22)]
        )
        self.assertEqual(reveal(dataset.column(s(0))), [0, 10, 20])
        self.assertEqual(reveal(dataset.column(s(1))), [1, 11, 21])
        self.assertEqual(reveal(dataset.column(s(2))), [2, 12, 22])

    def test_reveal_all_rows(self):
        dataset = ObliviousDataset(
            [s(0),  s(1),  s(2)],
            [s(10), s(11), s(12)],
            [s(20), s(21), s(22)]
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
            [s(0),  s(1),  s(2)],
            [s(10), s(11), s(12)],
            [s(20), s(21), s(22)]
        ).subset([s(0), s(0), s(0)])
        self.assertEqual(
            reveal(dataset),
            []
        )

    def test_subset(self):
        dataset = ObliviousDataset(
            [s(0),  s(1),  s(2)],
            [s(10), s(11), s(12)],
            [s(20), s(21), s(22)]
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
            [s(0),  s(1),  s(2)],
            [s(10), s(11), s(12)],
            [s(20), s(21), s(22)]
        ) \
            .subset([s(1), s(0), s(1)]) \
            .subset([s(0), s(1), s(1)])
        self.assertEqual(
            reveal(dataset),
            [
                [20, 21, 22]
            ]
        )


def reveal(secret):
    return mpc.run(output(secret))
