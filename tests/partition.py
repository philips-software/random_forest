import unittest
from mpyc.runtime import mpc
from src.partition import partition_on
from src.dataset import ObliviousDataset

s = mpc.SecInt()
output = mpc.output
run = mpc.run


class PartitionTests(unittest.TestCase):
    def test_partition_on(self):
        samples = ObliviousDataset(
            [s(0), s(1), s(1), s(0)],
            [s(1), s(0), s(1), s(1)],
            [s(0), s(0), s(0), s(1)],
        )
        is_active = [s(1), s(1), s(1)]
        best_attribute = s(2)
        best_attribute_value = s(0)
        left_active, right_active = partition_on(
            samples, is_active, best_attribute, best_attribute_value)
        self.assertEqual(run(output(left_active)), [0, 0, 1])
        self.assertEqual(run(output(right_active)), [1, 1, 0])

    def test_partition_on_partial_dataset(self):
        samples = ObliviousDataset(
            [s(0), s(1), s(1), s(0)],
            [s(1), s(0), s(1), s(1)],
            [s(0), s(0), s(0), s(1)],
        )
        is_active = [s(1), s(0), s(0)]
        best_attribute = s(2)
        best_attribute_value = s(0)
        left_active, right_active = partition_on(
            samples, is_active, best_attribute, best_attribute_value)
        self.assertEqual(run(output(left_active)), [0, 0, 0])
        self.assertEqual(run(output(right_active)), [1, 0, 0])
