import unittest
from src.gini import gini_impurity


class GiniTest(unittest.TestCase):

    def test_empty_attribute(self):
        self.assertEqual(gini_impurity(0, 0), 1)

    def test_balanced_attribute(self):
        self.assertEqual(gini_impurity(3, 3), 0.5)

    def test_homogeneous_attribute(self):
        self.assertEqual(gini_impurity(3, 0), 0)
        self.assertEqual(gini_impurity(0, 3), 0)
