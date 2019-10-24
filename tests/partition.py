import unittest
from mpyc.runtime import mpc
from src.output import output
from src.partition import partition_on
from src.dataset import ObliviousDataset, Sample

s = mpc.SecInt()
run = mpc.run


def sample(*inputs):
    return Sample(inputs, s(0))


class PartitionTests(unittest.TestCase):
    def test_partition_on(self):
        data = ObliviousDataset(
            sample(s(0), s(1), s(1), s(0)),
            sample(s(1), s(0), s(1), s(1)),
            sample(s(0), s(0), s(0), s(1))
        )
        left, right = partition_on(data, attribute_index=s(2), threshold=s(0))
        self.assertEqual(
            run(output(left)),
            [
                Sample([0, 0, 0, 1], 0)
            ]
        )
        self.assertEqual(
            run(output(right)),
            [
                Sample([0, 1, 1, 0], 0),
                Sample([1, 0, 1, 1], 0)
            ]
        )

    def test_partition_on_partial_dataset(self):
        data = ObliviousDataset(
            sample(s(0), s(1), s(1), s(0)),
            sample(s(1), s(0), s(1), s(1)),
            sample(s(0), s(0), s(0), s(1))
        ).select([s(1), s(0), s(0)])
        left, right = partition_on(data, attribute_index=s(2), threshold=s(0))
        self.assertEqual(run(output(left)), [])
        self.assertEqual(run(output(right)), [Sample([0, 1, 1, 0], 0)])
