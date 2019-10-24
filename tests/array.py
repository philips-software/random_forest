import unittest
from src.array import ObliviousArray
from mpyc.runtime import mpc
from src.output import output
import operator

s = mpc.SecInt()


class ObliviousArrayTest(unittest.TestCase):

    def test_initialize_with_list(self):
        self.assertEqual(
            reveal(ObliviousArray(s(1), s(2))),
            reveal(ObliviousArray([s(1), s(2)]))
        )
        self.assertEqual(
            reveal(ObliviousArray(s(1))),
            reveal(ObliviousArray([s(1)]))
        )

    def test_oblivious_indexing(self):
        array = ObliviousArray(s(10), s(20), s(30))
        self.assertEqual(reveal(array[s(0)]), 10)
        self.assertEqual(reveal(array[s(1)]), 20)
        self.assertEqual(reveal(array[s(2)]), 30)

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

    def test_select_with_oblivious_array_argument(self):
        array = ObliviousArray(s(10), s(20), s(30))
        included = ObliviousArray(s(1), s(0), s(1))
        array = array.select(included)
        self.assertEqual(reveal(array), [10, 30])

    def test_select_with_selection_as_argument(self):
        array = ObliviousArray(s(10), s(20), s(30))
        included = ObliviousArray(s(1), s(0), s(1))
        included = included.select(s(0), s(0), s(1))
        array = array.select(included)
        self.assertEqual(reveal(array), [30])

    def test_map(self):
        array = ObliviousArray(s(10), s(20), s(30))
        array = array.map(lambda x: 2 * x)
        self.assertEqual(reveal(array), [20, 40, 60])

    def test_map_on_selected_elements(self):
        array = ObliviousArray(s(10), s(20), s(30))
        array = array.select(s(1), s(0), s(1))
        array = array.map(lambda x: 2 * x)
        self.assertEqual(reveal(array), [20, 60])

    def test_reduce(self):
        array = ObliviousArray(s(10), s(20), s(30))
        product = array.reduce(s(1), operator.mul)
        self.assertEqual(reveal(product), 6000)

    def test_reduce_on_selected_elements(self):
        array = ObliviousArray(s(10), s(20), s(30))
        array = array.select(s(1), s(0), s(1))
        product = array.reduce(s(1), operator.mul)
        self.assertEqual(reveal(product), 300)

    def test_sum(self):
        array = ObliviousArray(s(10), s(20), s(30))
        sum = array.sum()
        self.assertEqual(reveal(sum), 60)

    def test_sum_on_selected_elements(self):
        array = ObliviousArray(s(10), s(20), s(30))
        array = array.select(s(1), s(0), s(1))
        sum = array.sum()
        self.assertEqual(reveal(sum), 40)


def reveal(secret):
    return mpc.run(output(secret))
