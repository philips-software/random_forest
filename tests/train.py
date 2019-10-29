import unittest
from mpyc.runtime import mpc
from src.dataset import ObliviousDataset, Sample
from src.train import train
from src.tree import Node
from src.output import output

s = mpc.SecInt()


class TrainTests(unittest.TestCase):

    def test_single_sample(self):
        samples = ObliviousDataset(Sample([s(1)], s(1)))
        self.assertEqual(reveal(train(samples)), Node(0))

    def test_two_samples_two_attributes(self):
        samples = ObliviousDataset(
            Sample([s(1), s(0)], s(0)),
            Sample([s(1), s(1)], s(1)))
        self.assertEqual(reveal(train(samples)), Node(1))


def reveal(secret):
    return mpc.run(output(secret))
