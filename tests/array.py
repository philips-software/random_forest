import unittest
from src.array import ObliviousArray
from mpyc.runtime import mpc

s = mpc.SecInt()


class ObliviousArrayTest(unittest.TestCase):
    def test_oblivious_indexing(self):
        array = ObliviousArray(s(10), s(20), s(30))
        self.assertEquals(reveal(array[s(0)]), 10)
        self.assertEquals(reveal(array[s(1)]), 20)
        self.assertEquals(reveal(array[s(2)]), 30)


def reveal(secret):
    return mpc.run(mpc.output(secret))
