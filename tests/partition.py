import unittest

from src.dataset import ObliviousDataset, Sample
from src.partition import partition_on, partition_on_binary_attribute
from src.secint import secint as s
from tests.reveal import reveal


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

    def test_partition_on_partial_dataset(self):
        data = ObliviousDataset(
            sample(s(0), s(1), s(1), s(0)),
            sample(s(1), s(0), s(1), s(1)),
            sample(s(0), s(0), s(0), s(1))
        ).select([s(1), s(0), s(0)])
        left, right = partition_on(data, attribute_index=s(2), threshold=s(0))
        self.assertEqual(reveal(left), [])
        self.assertEqual(reveal(right), [Sample([0, 1, 1, 0], 0)])

    def test_partition_on_binary_attribute(self):
        data = ObliviousDataset(
            sample(s(0), s(1), s(1), s(0)),
            sample(s(1), s(0), s(1), s(1)),
            sample(s(0), s(0), s(0), s(1))
        )
        left, right = partition_on_binary_attribute(data, attribute_index=s(2))
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
