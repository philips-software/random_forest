import unittest

from src.secint import secint as s
from src.tree import Node
from tests.reveal import reveal


class TreeTest(unittest.TestCase):
    def test_reveal_single_node(self):
        self.assertEqual(reveal(Node(s(1))), Node(1))
