import unittest
from src.array import ObliviousArray
from mpyc.runtime import mpc
from src.output import output

s = mpc.SecInt()


class ObliviousArrayTest(unittest.TestCase):
    def test_oblivious_indexing(self):
        array = ObliviousArray(s(10), s(20), s(30))
        self.assertEquals(reveal(array[s(0)]), 10)
        self.assertEquals(reveal(array[s(1)]), 20)
        self.assertEquals(reveal(array[s(2)]), 30)

    def test_reveal_all_elements(self):
        array = ObliviousArray(s(10), s(20), s(30))
        self.assertEqual(reveal(array), [10, 20, 30])

    def test_reveal_selected_elements(self):
        array = ObliviousArray(s(10), s(20), s(30))
        array = array.select(s(0), s(1), s(0))
        self.assertEqual(reveal(array), [20])

    def test_select_and_select_again(self):
        array = ObliviousArray(s(10), s(20), s(30))
        array = array.select(s(1), s(0), s(1))
        array = array.select(s(0), s(1), s(1))
        self.assertEqual(reveal(array), [30])


def reveal(secret):
    return mpc.run(output(secret))
