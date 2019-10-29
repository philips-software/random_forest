import unittest

from src.dataset import ObliviousDataset, Sample
from src.partition import partition_continuous, partition_binary
from src.secint import secint as s
from tests.reveal import reveal


def sample(*inputs):
    return Sample(inputs, s(0))


class PartitionTests(unittest.TestCase):
    def test_partition_on_continuous_attribute(self):
        data = ObliviousDataset(
            sample(s(1), s(3), s(3), s(1)),
            sample(s(3), s(1), s(3), s(3)),
            sample(s(1), s(1), s(1), s(3))
        )
        left, right = partition_continuous(
            data, attribute_index=s(2), threshold=s(2))
        self.assertEqual(
            reveal(left),
            [
                Sample([1, 1, 1, 3], 0)
            ]
        )
        self.assertEqual(
            reveal(right),
            [
                Sample([1, 3, 3, 1], 0),
                Sample([3, 1, 3, 3], 0)
            ]
        )

    def test_partition_on_continuous_attribute_partial_dataset(self):
        data = ObliviousDataset(
            sample(s(1), s(3), s(3), s(1)),
            sample(s(3), s(1), s(3), s(3)),
            sample(s(1), s(1), s(1), s(3))
        ).select([s(1), s(0), s(0)])
        left, right = partition_continuous(
            data, attribute_index=s(2), threshold=s(2))
        self.assertEqual(reveal(left), [])
        self.assertEqual(reveal(right), [Sample([1, 3, 3, 1], 0)])

    def test_partition_on_binary_attribute(self):
        data = ObliviousDataset(
            sample(s(0), s(1), s(1), s(0)),
            sample(s(1), s(0), s(1), s(1)),
            sample(s(0), s(0), s(0), s(1))
        )
        left, right = partition_binary(data, attribute_index=s(2))
        self.assertEqual(
            reveal(left),
            [
                Sample([0, 0, 0, 1], 0)
            ]
        )
        self.assertEqual(
            reveal(right),
            [
                Sample([0, 1, 1, 0], 0),
                Sample([1, 0, 1, 1], 0)
            ]
        )
