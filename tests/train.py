import unittest
from src.dataset import ObliviousDataset, Sample
from src.train import train
from src.tree import Node
from tests.reveal import reveal
from src.secint import secint as s


class TrainTests(unittest.TestCase):

    def test_single_sample(self):
        samples = ObliviousDataset(Sample([s(1)], s(1)))
        self.assertEqual(reveal(train(samples)), Node(0))

    def test_two_samples_two_attributes(self):
        samples = ObliviousDataset(
            Sample([s(1), s(0)], s(0)),
            Sample([s(1), s(1)], s(1)))
        self.assertEqual(reveal(train(samples)), Node(1))

    def test_single_sample_with_some_depth(self):
        samples = ObliviousDataset(Sample([s(1)], s(1)))
        self.assertEqual(reveal(train(samples, depth=2)),
                         Node(0, left=Node(0), right=Node(0)))

    def test_multiple_samples_with_some_depth(self):
        samples = ObliviousDataset(
            Sample([s(0), s(1)], s(0)),
            Sample([s(1), s(0)], s(0)),
            Sample([s(1), s(0)], s(0)),
            Sample([s(1), s(1)], s(1)))
        self.assertEqual(reveal(train(samples, depth=2)),
                         Node(1, left=Node(1), right=Node(0)))
