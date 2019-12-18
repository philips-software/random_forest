import unittest

from src.dataset import ObliviousDataset, Sample
from src.secint import secint as s
from tests.reveal import reveal


class SortedDatasetTests(unittest.TestCase):

    def test_it_is_a_dataset(self):
        dataset = ObliviousDataset.create()
        sorted = dataset.sort()
        self.assertIsInstance(sorted, ObliviousDataset)

    def test_it_has_sorted_columns(self):
        dataset = ObliviousDataset.create(
            Sample([s(3), s(6)], s(0)),
            Sample([s(2), s(4)], s(0)),
            Sample([s(1), s(5)], s(0))
        )
        sorted = dataset.sort()
        self.assertEqual(reveal(sorted.sorted_column(0)), [1, 2, 3])
        self.assertEqual(reveal(sorted.sorted_column(1)), [4, 5, 6])
