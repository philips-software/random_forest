import unittest
from src.gini import *


class GiniTest(unittest.TestCase):

    def test_empty_attribute(self):
        self.assertEqual(gini_impurity(0, 0), 1)

    def test_balanced_attribute(self):
        self.assertEqual(gini_impurity(3, 3), 0.5)

    def test_homogeneous_attribute(self):
        self.assertEqual(gini_impurity(3, 0), 0)
        self.assertEqual(gini_impurity(0, 3), 0)

    def test_gini_gain_single_value(self):
        numerator, denominator = gini_gain_scaled_quotient(1, 0, 1, 0, 0, 0)
        total = 1
        gain = (1 / total) * numerator / denominator
        self.assertEqual(gain, 1)

    def test_gini_gain_uniform(self):
        numerator, denominator = gini_gain_scaled_quotient(2, 2, 1, 1, 1, 1)
        total = 4
        gain = (1 / total) * (numerator / denominator)
        self.assertEqual(gain, 0.5)

    def test_gini_gain_perfect_split(self):
        numerator, denominator = gini_gain_scaled_quotient(2, 2, 2, 0, 0, 2)
        total = 4
        gain = (1 / total) * (numerator / denominator)
        self.assertEqual(gain, 1)
