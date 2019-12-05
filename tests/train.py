import unittest
from src.dataset import ObliviousDataset, Sample
from src.train import train
from src.tree import Branch, Leaf
from tests.reveal import reveal
from src.secint import secint as s


class TrainTests(unittest.TestCase):

    def test_single_sample_depth_zero_outcome_1(self):
        samples = ObliviousDataset.create(Sample([s(1)], s(1)))
        self.assertEqual(reveal(train(samples, depth=0)), leaf(1))

    def test_single_sample_depth_zero_outcome_0(self):
        samples = ObliviousDataset.create(Sample([s(1)], s(0)))
        self.assertEqual(reveal(train(samples, depth=0)), leaf(0))

    def test_two_samples_two_attributes(self):
        samples = ObliviousDataset.create(
            Sample([s(1), s(0)], s(0)),
            Sample([s(1), s(1)], s(1)))
        self.assertEqual(
            reveal(train(samples, depth=1)),
            Branch(1, threshold=0, left=leaf(0), right=leaf(1)))

    def test_single_sample_depth_one(self):
        samples = ObliviousDataset.create(Sample([s(1)], s(1)))
        self.assertEqual(
            reveal(train(samples, depth=1)),
            Branch(0, threshold=0, left=pruned(), right=leaf(1)))

    def test_single_sample_with_some_depth(self):
        samples = ObliviousDataset.create(Sample([s(1)], s(1)))
        self.assertEqual(
            reveal(train(samples, depth=2)),
            Branch(0, threshold=0,
                   left=Branch(0, threshold=0, left=pruned(), right=pruned()),
                   right=Branch(0, threshold=0, left=pruned(), right=leaf(1))))

    def test_multiple_samples_with_some_depth(self):
        samples = ObliviousDataset.create(
            Sample([s(0), s(1)], s(0)),
            Sample([s(1), s(0)], s(0)),
            Sample([s(1), s(0)], s(0)),
            Sample([s(1), s(1)], s(1)))
        self.assertEqual(
            reveal(train(samples, depth=2)),
            Branch(1,
                   threshold=0,
                   left=Branch(
                       1,  # random, could have been zero as well
                       threshold=0,
                       left=leaf(0),
                       right=pruned()
                   ),
                   right=Branch(0, threshold=0, left=leaf(0), right=leaf(1))))

    def test_continuous_attributes(self):
        samples = ObliviousDataset.create(
            Sample([s(1), s(2)], s(0)),
            Sample([s(1), s(3)], s(1)),
            continuous=[False, True])
        self.assertEqual(
            reveal(train(samples, depth=1)),
            Branch(1, threshold=2, left=leaf(0), right=leaf(1)))


def leaf(outcome):
    return Leaf(outcome, False)


def pruned():
    return Leaf(0, True)
