import unittest
from src.tree import Branch, Leaf
from src.secint import secint as s
from tests.reveal import reveal
from src.classification import classify


class ClassificationTest(unittest.TestCase):

    def test_classify_with_only_leaf_node(self):
        sample = [s(1), s(0), s(1)]
        tree = leaf(s(1))
        self.assertEqual(reveal(classify(sample, tree)), 1)

    def test_classify_with_a_branch(self):
        tree = Branch(s(1), s(0), leaf(s(1)), leaf(s(0)))
        self.assertEqual(reveal(classify([s(1), s(0), s(1)], tree)), 1)
        self.assertEqual(reveal(classify([s(1), s(1), s(1)], tree)), 0)

    def test_classify_with_pruned_leaf(self):
        tree = Branch(s(1), s(0), leaf(s(1)), pruned())
        self.assertEqual(reveal(classify([s(1), s(0), s(1)], tree)), 1)
        self.assertEqual(reveal(classify([s(1), s(1), s(1)], tree)), 1)

    def test_classify_with_pruned_subtree(self):
        tree = Branch(s(1), s(0),
                      Branch(s(0), s(0),
                             pruned(),
                             pruned()),
                      Branch(s(2), s(0),
                             leaf(s(0)),
                             leaf(s(1))))
        self.assertEqual(reveal(classify([s(0), s(0), s(1)], tree)), 1)

    def test_classify_with_continuous_attribute(self):
        tree = Branch(s(1), s(2), leaf(s(1)), leaf(s(0)))
        self.assertEqual(reveal(classify([s(1), s(2), s(1)], tree)), 1)
        self.assertEqual(reveal(classify([s(1), s(3), s(1)], tree)), 0)


def leaf(outcome):
    return Leaf(outcome, s(False))


def pruned():
    return Leaf(s(0), s(True))
