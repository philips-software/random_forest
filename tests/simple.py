import unittest
from mpyc.runtime import mpc

secint = mpc.SecInt()
input = mpc.input
output = mpc.output
run = mpc.run


class SimpleTest(unittest.TestCase):

    def test_mpc(self):
        a = secint(40)
        b = secint(2)
        c = run(output(a + b))
        self.assertEqual(c, 42)
