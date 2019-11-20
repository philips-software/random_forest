import unittest
from src.dataset import Sample
from src.tree import Leaf
from src.secint import secint as s
from tests.reveal import reveal
from src.classification import classify


class ClassificationTest(unittest.TestCase):

    def test_classify_with_only_leaf_node(self):
        sample = [s(1), s(0), s(1)]
        tree = Leaf(s(1), pruned=False)
        self.assertEqual(reveal(classify(sample, tree)), 1)
