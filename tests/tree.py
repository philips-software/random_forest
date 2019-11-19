import unittest

from src.secint import secint as s
from src.tree import Branch, Leaf
from tests.reveal import reveal


class TreeTest(unittest.TestCase):
    def test_reveal_leaf(self):
        self.assertEqual(reveal(Leaf(s(1), pruned=s(False))), Leaf(1, False))

    def test_reveal_branches(self):
        tree = Branch(s(0),
                      left=Leaf(s(1), pruned=s(False)),
                      right=Leaf(s(2), pruned=s(True)))
        expected_output = Branch(0,
                                 left=Leaf(1, pruned=False),
                                 right=Leaf(2, pruned=True))
        self.assertEqual(reveal(tree), expected_output)
