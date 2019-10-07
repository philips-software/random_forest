import unittest
from src.gini import gini_impurity


class GiniTest(unittest.TestCase):

    def test_empty_matrix(self):
        self.assertEqual(gini_impurity([]), 1)
