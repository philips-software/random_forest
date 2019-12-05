import unittest
from src.tree import Leaf, Branch
from src.forest import train_forest, bootstrap, random_attributes
from tests.example import binary_samples as samples
from tests.reveal import reveal
from tests.async_test import async_test


class ForestTest(unittest.TestCase):

    @async_test
    async def test_forest_consists_of_multiple_trees(self):
        self.assertEqual(len(await train_forest(samples, 2, 1)), 2)
        self.assertEqual(len(await train_forest(samples, 3, 1)), 3)

    @async_test
    async def test_each_tree_has_same_depth(self):
        tree_depth = 2
        forest = await train_forest(samples, 3, tree_depth)
        for tree in forest:
            self.assertEqual(depth(tree), tree_depth)

    @async_test
    async def test_trees_are_different(self):
        trees = [reveal(tree) for tree in await train_forest(samples, 3, 2)]
        self.assertTrue(trees[0] != trees[1] or trees[0] != trees[2])

    def test_bootstrap_selection_is_random(self):
        selection1 = reveal(bootstrap(samples))
        selection2 = reveal(bootstrap(samples))
        self.assertNotEqual(selection1, selection2)

    def test_bootstrap_selects_from_the_original_samples(self):
        selection = reveal(bootstrap(samples))
        all = reveal(samples)
        for sample in selection:
            self.assertIn(sample, all)

    def test_attribute_selection_is_random(self):
        selection1 = reveal(random_attributes(samples, 3))
        selection2 = reveal(random_attributes(samples, 3))
        self.assertNotEqual(selection1, selection2)

    def test_features_are_selected_from_the_original_sample(self):
        selection = random_attributes(samples, 3)
        selected_columns = reveal(columns(selection))
        all_columns = reveal(columns(samples))
        for column in selected_columns:
            self.assertIn(column, all_columns)


def depth(tree):
    if isinstance(tree, Leaf):
        return 0
    if isinstance(tree, Branch):
        return depth(tree.left) + 1


def columns(samples):
    return [
        [samples[row][column] for row in range(len(samples))]
        for column in range(samples.number_of_attributes)
    ]
