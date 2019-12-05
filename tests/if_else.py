import unittest

from src.if_else import if_else
from src.secint import secint as s
from tests.reveal import reveal


class IfElseTest(unittest.TestCase):
    def test_tuple(self):
        chosen = if_else(s(True), (s(1), s(2)), (s(3), s(4)))
        self.assertEqual(reveal(chosen), (1, 2))
