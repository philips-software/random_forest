import unittest
from src.dataset import ObliviousDataset, Sample
from src.train import train
from src.tree import Branch, Leaf
from tests.reveal import reveal
from src.secint import secint as s


class TrainTests(unittest.TestCase):

    def test_single_sample_depth_zero_outcome_1(self):
        samples = ObliviousDataset.create(Sample([s(1)], s(1)))
        self.assertEqual(reveal(train(samples, depth=0)), Leaf(1, False))

    def test_single_sample_depth_zero_outcome_0(self):
        samples = ObliviousDataset.create(Sample([s(1)], s(0)))
        self.assertEqual(reveal(train(samples, depth=0)), Leaf(0, False))

    def test_two_samples_two_attributes(self):
        samples = ObliviousDataset.create(
            Sample([s(1), s(0)], s(0)),
            Sample([s(1), s(1)], s(1)))
        self.assertEqual(
            reveal(train(samples, depth=1)),
            Branch(1, left=Leaf(0, False), right=Leaf(1, False)))

    def test_single_sample_depth_one(self):
        samples = ObliviousDataset.create(Sample([s(1)], s(1)))
        self.assertEqual(
            reveal(train(samples, depth=1)),
            Branch(0,
                   left=Leaf(0, pruned=True),
                   right=Leaf(1, pruned=False)))

    def test_single_sample_with_some_depth(self):
        samples = ObliviousDataset.create(Sample([s(1)], s(1)))
        self.assertEqual(
            reveal(train(samples, depth=2)),
            Branch(0,
                   left=Branch(
                       0, left=Leaf(0, True), right=Leaf(0, True)
                   ),
                   right=Branch(
                       0, left=Leaf(0, True), right=Leaf(1, False)
                   )))

    def test_multiple_samples_with_some_depth(self):
        samples = ObliviousDataset.create(
            Sample([s(0), s(1)], s(0)),
            Sample([s(1), s(0)], s(0)),
            Sample([s(1), s(0)], s(0)),
            Sample([s(1), s(1)], s(1)))
        self.assertEqual(
            reveal(train(samples, depth=2)),
            Branch(1,
                   left=Branch(
                       1,  # random, could have been zero as well
                       left=Leaf(0, False),
                       right=Leaf(0, True)
                   ),
                   right=Branch(
                       0, left=Leaf(0, False), right=Leaf(1, False)
                   )))
