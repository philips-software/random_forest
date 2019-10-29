import unittest

from src.secint import secint as s
from src.tree import Node
from tests.reveal import reveal


class TreeTest(unittest.TestCase):
    def test_reveal_single_node(self):
        self.assertEqual(reveal(Node(s(1))), Node(1))

    def test_reveal_branches(self):
        tree = Node(s(0), left=Node(s(1)), right=Node(s(2)))
        expected_output = Node(0, left=Node(1), right=Node(2))
        self.assertEqual(reveal(tree), expected_output)
