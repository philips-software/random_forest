import unittest
from src.dataset import ObliviousDataset
from src.tree import Leaf, Branch
from src.forest import train_forest
from tests.example import samples
from tests.reveal import reveal


class ForestTest(unittest.TestCase):

    def test_forest_consists_of_multiple_trees(self):
        self.assertEqual(len(train_forest(samples, 2, 1)), 2)
        self.assertEqual(len(train_forest(samples, 3, 1)), 3)

    def test_each_tree_has_same_depth(self):
        tree_depth = 2
        forest = train_forest(samples, 3, tree_depth)
        for tree in forest:
            self.assertEqual(depth(tree), tree_depth)

    def test_trees_are_different(self):
        trees = [reveal(tree) for tree in train_forest(samples, 3, 2)]
        self.assertTrue(trees[0] != trees[1] or trees[0] != trees[2])


def depth(tree):
    if isinstance(tree, Leaf):
        return 0
    if isinstance(tree, Branch):
        return depth(tree.left) + 1
