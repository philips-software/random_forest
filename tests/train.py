import unittest
from src.dataset import ObliviousDataset, Sample
from src.train import train
from src.tree import Branch
from tests.reveal import reveal
from src.secint import secint as s


class TrainTests(unittest.TestCase):

    def test_single_sample(self):
        samples = ObliviousDataset.create(Sample([s(1)], s(1)))
        self.assertEqual(reveal(train(samples, depth=1)), Branch(0))

    def test_two_samples_two_attributes(self):
        samples = ObliviousDataset.create(
            Sample([s(1), s(0)], s(0)),
            Sample([s(1), s(1)], s(1)))
        self.assertEqual(reveal(train(samples, depth=1)), Branch(1))

    def test_single_sample_with_some_depth(self):
        samples = ObliviousDataset.create(Sample([s(1)], s(1)))
        self.assertEqual(reveal(train(samples, depth=2)),
                         Branch(0, left=Branch(0), right=Branch(0)))

    def test_multiple_samples_with_some_depth(self):
        samples = ObliviousDataset.create(
            Sample([s(0), s(1)], s(0)),
            Sample([s(1), s(0)], s(0)),
            Sample([s(1), s(0)], s(0)),
            Sample([s(1), s(1)], s(1)))
        self.assertEqual(reveal(train(samples, depth=2)),
                         Branch(1, left=Branch(1), right=Branch(0)))
