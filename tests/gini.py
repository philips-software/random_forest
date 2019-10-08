import unittest
from src.gini import *


class GiniTest(unittest.TestCase):

    def test_gini_gain_uniform(self):
        numerator, denominator = gini_gain_quotient(2, 2, 1, 1, 1, 1)
        total = 4
        gain = (1 / total) * (numerator / denominator)
        self.assertEqual(gain, 0.5)

    def test_gini_gain_perfect_split(self):
        numerator, denominator = gini_gain_quotient(2, 2, 2, 0, 0, 2)
        total = 4
        gain = (1 / total) * (numerator / denominator)
        self.assertEqual(gain, 1)

    def test_avoidance_of_division_by_zero(self):
        numerator, denominator = (gini_gain_quotient(0, 0, 0, 0, 0, 0))
        total = 1
        gain = (1 / total) * numerator / avoid_zero(denominator)
        self.assertEqual(gain, 0)
