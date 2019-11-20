import unittest
from src.dataset import Sample
from src.tree import Branch, Leaf
from src.secint import secint as s
from tests.reveal import reveal
from src.classification import classify


class ClassificationTest(unittest.TestCase):

    def test_classify_with_only_leaf_node(self):
        sample = [s(1), s(0), s(1)]
        tree = Leaf(s(1), pruned=False)
        self.assertEqual(reveal(classify(sample, tree)), 1)

    def test_classify_with_a_branch(self):
        tree = Branch(s(1),
                      left=Leaf(s(1), s(False)),
                      right=Leaf(s(0), s(False)))
        self.assertEqual(reveal(classify([s(1), s(0), s(1)], tree)), 1)
        self.assertEqual(reveal(classify([s(1), s(1), s(1)], tree)), 0)

    def test_classify_with_pruned_leaf(self):
        tree = Branch(s(1),
                      left=Leaf(s(1), s(True)),
                      right=Leaf(s(0), s(False)))
        self.assertEqual(reveal(classify([s(1), s(0), s(1)], tree)), 0)
        self.assertEqual(reveal(classify([s(1), s(1), s(1)], tree)), 0)
