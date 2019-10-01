import unittest


class SimpleTest(unittest.TestCase):

    def test_fails(self):
        self.assertEqual(1, 2)
