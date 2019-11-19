import unittest

from src.secint import secint as s
from src.tree import Branch
from tests.reveal import reveal


class TreeTest(unittest.TestCase):
    def test_reveal_single_node(self):
        self.assertEqual(reveal(Branch(s(1))), Branch(1))

    def test_reveal_branches(self):
        tree = Branch(s(0), left=Branch(s(1)), right=Branch(s(2)))
        expected_output = Branch(0, left=Branch(1), right=Branch(2))
        self.assertEqual(reveal(tree), expected_output)
