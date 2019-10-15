import unittest
import src.dataset
from mpyc.runtime import mpc

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

    def test_all_rows_are_active(self):
        dataset = ObliviousDataset(
            [s(0),  s(1),  s(2)],
            [s(10), s(11), s(12)],
            [s(20), s(21), s(22)]
        )
        self.assertEqual(
            dataset.reveal(),
            [
                [0,  1,  2],
                [10, 11, 12],
                [20, 21, 22]
            ]
        )

    def test_no_rows_are_active(self):
        dataset = ObliviousDataset(
            [s(0),  s(1),  s(2)],
            [s(10), s(11), s(12)],
            [s(20), s(21), s(22)]
        )
        dataset.active_rows = [s(0), s(0), s(0)]
        self.assertEqual(
            dataset.reveal(),
            [])


def reveal(secret):
    return mpc.run(mpc.output(secret))


class ObliviousDataset(src.dataset.ObliviousDataset):
    def reveal(self):
        all_rows = list(map(reveal, self.rows))
        active_rows = reveal(self.active_rows)
        return list(map(
            first,
            filter(
                second,
                zip(all_rows, active_rows))))


def first(pair): return pair[0]
def second(pair): return pair[1]
