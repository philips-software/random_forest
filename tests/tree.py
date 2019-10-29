import unittest
from src.tree import Node
from mpyc.runtime import mpc
from tests.reveal import reveal

s = mpc.SecInt()


class TreeTest(unittest.TestCase):
    def test_reveal_single_node(self):
        self.assertEqual(reveal(Node(s(1))), Node(1))
