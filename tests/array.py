import unittest
import operator

from src.array import ObliviousArray
from src.secint import secint as s
from tests.reveal import reveal


class ObliviousArrayTest(unittest.TestCase):

    def test_initialize_with_list(self):
        self.assertEqual(
            reveal(ObliviousArray.create(s(1), s(2))),
            reveal(ObliviousArray([s(1), s(2)]))
        )
        self.assertEqual(
            reveal(ObliviousArray.create(s(1))),
            reveal(ObliviousArray([s(1)]))
        )

    def test_len(self):
        self.assertEqual(reveal(ObliviousArray.create().len()), 0)
        self.assertEqual(reveal(ObliviousArray(
            [s(0), s(1), s(2)]).len()), 3)

    def test_len_of_subset(self):
        dataset = ObliviousArray.create(s(0), s(1), s(2))
        dataset = dataset.select(s(1), s(0), s(1))
        self.assertEqual(reveal(dataset.len()), 2)

    def test_reveal_all_elements(self):
        array = ObliviousArray.create(s(10), s(20), s(30))
        self.assertEqual(reveal(array), [10, 20, 30])

    def test_reveal_selected_elements(self):
        array = ObliviousArray.create(s(10), s(20), s(30))
        array = array.select(s(0), s(1), s(0))
        self.assertEqual(reveal(array), [20])

    def test_select_and_select_again(self):
        array = ObliviousArray.create(s(10), s(20), s(30))
        array = array.select(s(1), s(0), s(1))
        array = array.select(s(0), s(1), s(1))
        self.assertEqual(reveal(array), [30])

    def test_select_with_oblivious_array_argument(self):
        array = ObliviousArray.create(s(10), s(20), s(30))
        included = ObliviousArray.create(s(1), s(0), s(1))
        array = array.select(included)
        self.assertEqual(reveal(array), [10, 30])

    def test_select_with_selection_as_argument(self):
        array = ObliviousArray.create(s(10), s(20), s(30))
        included = ObliviousArray.create(s(1), s(0), s(1))
        included = included.select(s(0), s(0), s(1))
        array = array.select(included)
        self.assertEqual(reveal(array), [30])

    def test_map(self):
        array = ObliviousArray.create(s(10), s(20), s(30))
        array = array.map(lambda x: 2 * x)
        self.assertEqual(reveal(array), [20, 40, 60])

    def test_map_on_selected_elements(self):
        array = ObliviousArray.create(s(10), s(20), s(30))
        array = array.select(s(1), s(0), s(1))
        array = array.map(lambda x: 2 * x)
        self.assertEqual(reveal(array), [20, 60])

    def test_reduce(self):
        array = ObliviousArray.create(s(10), s(20), s(30))
        product = array.reduce(s(1), operator.mul)
        self.assertEqual(reveal(product), 6000)

    def test_reduce_on_selected_elements(self):
        array = ObliviousArray.create(s(10), s(20), s(30))
        array = array.select(s(1), s(0), s(1))
        product = array.reduce(s(1), operator.mul)
        self.assertEqual(reveal(product), 300)

    def test_reduce_with_initial_value(self):
        array = ObliviousArray.create()
        product = array.reduce(s(1), operator.mul, initial=s(42))
        self.assertEqual(reveal(product), 42)

    def test_sum(self):
        array = ObliviousArray.create(s(10), s(20), s(30))
        sum = array.sum()
        self.assertEqual(reveal(sum), 60)

    def test_sum_on_selected_elements(self):
        array = ObliviousArray.create(s(10), s(20), s(30))
        array = array.select(s(1), s(0), s(1))
        sum = array.sum()
        self.assertEqual(reveal(sum), 40)

    def test_getitem_with_secret_index(self):
        array = ObliviousArray.create(s(10), s(20), s(30))
        item = array.getitem(s(1))
        self.assertEqual(reveal(item), 20)

    def test_getitem_on_selected_elements(self):
        array = ObliviousArray.create(s(10), s(20), s(30))
        selection = array.select(s(1), s(0), s(1))
        self.assertEqual(reveal(selection.getitem(s(0))), 10)
        self.assertEqual(reveal(selection.getitem(s(1))), 0)
        self.assertEqual(reveal(selection.getitem(s(2))), 30)
