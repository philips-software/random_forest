import unittest
from src.dataset import ObliviousDataset
from mpyc.runtime import mpc

s = mpc.SecInt()


class ObliviousDatasetTest(unittest.TestCase):
    def test_initialize_with_list(self):
        self.assertEquals(
            ObliviousDataset([1, 2], [3, 4]),
            ObliviousDataset([[1, 2], [3, 4]])
        )

    def test_column(self):
        dataset = ObliviousDataset(
            [s(0),  s(1),  s(2)],
            [s(10), s(11), s(12)],
            [s(20), s(21), s(22)]
        )
        self.assertEquals(reveal(dataset.column(s(0))), [0, 10, 20])
        self.assertEquals(reveal(dataset.column(s(1))), [1, 11, 21])
        self.assertEquals(reveal(dataset.column(s(2))), [2, 12, 22])


def reveal(secret):
    return mpc.run(mpc.output(secret))
