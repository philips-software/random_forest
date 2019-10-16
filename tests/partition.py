import unittest
from mpyc.runtime import mpc
from src.output import output
from src.partition import partition_on
from src.dataset import ObliviousDataset

s = mpc.SecInt()
run = mpc.run


class PartitionTests(unittest.TestCase):
    def test_partition_on(self):
        data = ObliviousDataset(
            [s(0), s(1), s(1), s(0)],
            [s(1), s(0), s(1), s(1)],
            [s(0), s(0), s(0), s(1)],
        )
        left, right = partition_on(data, attribute_index=s(2), threshold=s(0))
        self.assertEqual(
            run(output(left)),
            [
                [0, 0, 0, 1]
            ]
        )
        self.assertEqual(
            run(output(right)),
            [
                [0, 1, 1, 0],
                [1, 0, 1, 1]
            ]
        )

    def test_partition_on_partial_dataset(self):
        data = ObliviousDataset(
            [s(0), s(1), s(1), s(0)],
            [s(1), s(0), s(1), s(1)],
            [s(0), s(0), s(0), s(1)],
        ).subset([s(1), s(0), s(0)])
        left, right = partition_on(data, attribute_index=s(2), threshold=s(0))
        self.assertEqual(run(output(left)), [])
        self.assertEqual(run(output(right)), [[0, 1, 1, 0]])
