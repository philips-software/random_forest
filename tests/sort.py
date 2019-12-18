import unittest

from src.dataset import ObliviousDataset, Sample
from src.secint import secint as s
from src.sort import sort
from tests.reveal import reveal


class SortTests(unittest.TestCase):
    def test_sorts_column_and_outcomes_of_array(self):
        dataset = ObliviousDataset.create(
            Sample([s(2)], s(5)),
            Sample([s(1)], s(6)),
            Sample([s(3)], s(7)),
            Sample([s(4)], s(8))
        )
        column = dataset.column(s(0))
        outcomes = dataset.outcomes
        sorted_column, sorted_outcomes, _ = sort(column, outcomes)
        self.assertEqual(reveal(sorted_column), [1, 2, 3, 4])
        self.assertEqual(reveal(sorted_outcomes), [6, 5, 7, 8])

    def test_sorts_column_and_outcomes_of_selection(self):
        dataset = ObliviousDataset.create(
            Sample([s(2)], s(5)),
            Sample([s(1)], s(6)),
            Sample([s(3)], s(7)),
            Sample([s(4)], s(8))
        ).select(s(1), s(1), s(1), s(0))
        column = dataset.column(s(0))
        outcomes = dataset.outcomes
        sorted_column, sorted_outcomes, _ = sort(column, outcomes)
        self.assertEqual(reveal(sorted_column), [1, 2, 3])
        self.assertEqual(reveal(sorted_outcomes), [6, 5, 7])
