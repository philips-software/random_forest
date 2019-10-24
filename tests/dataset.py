import unittest
from src.dataset import ObliviousDataset, Sample
from mpyc.runtime import mpc
from src.output import output

s = mpc.SecInt()


def sample(*inputs):
    return Sample(inputs, s(0))


class ObliviousDatasetTest(unittest.TestCase):
    def test_initialize_with_list(self):
        self.assertEqual(
            reveal(ObliviousDataset(sample(s(1), s(2)), sample(s(3), s(4)))),
            reveal(ObliviousDataset([sample(s(1), s(2)), sample(s(3), s(4))]))
        )
        self.assertEqual(
            reveal(ObliviousDataset(sample(s(1), s(2)))),
            reveal(ObliviousDataset([sample(s(1), s(2))]))
        )

    def test_len(self):
        self.assertEqual(len(ObliviousDataset()), 0)
        self.assertEqual(len(ObliviousDataset(
            [sample(0), sample(1), sample(2)])), 3)

    def test_row_indexing(self):
        data = ObliviousDataset(sample(s(0), s(1)), sample(s(2), s(3)))
        self.assertEqual(reveal(data[0]), Sample([0, 1], 0))
        self.assertEqual(reveal(data[1]), Sample([2, 3], 0))

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
                Sample([0,  1,  2], 0),
                Sample([10, 11, 12], 0),
                Sample([20, 21, 22], 0)
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
                Sample([0,  1,  2], 0),
                Sample([20, 21, 22], 0)
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
                Sample([20, 21, 22], 0)
            ]
        )

    def test_column_of_subset(self):
        dataset = ObliviousDataset(
            sample(s(0),  s(1),  s(2)),
            sample(s(10), s(11), s(12)),
            sample(s(20), s(21), s(22))
        ).subset([s(1), s(0), s(1)])
        self.assertEqual(reveal(dataset.column(s(1))), [1, 21])

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
