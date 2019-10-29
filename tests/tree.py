import unittest
from src.tree import Node
from mpyc.runtime import mpc
from src.output import output

s = mpc.SecInt()


class TreeTest(unittest.TestCase):
    def test_reveal_single_node(self):
        self.assertEqual(reveal(Node(s(1))), Node(1))


def reveal(secret):
    return mpc.run(output(secret))
