from src.tree import Branch, Leaf
from mpyc.runtime import mpc


def classify(sample, tree):
    return findleaf(sample, tree).outcome_class


def findleaf(sample, tree):
    if isinstance(tree, Branch):
        value = getitem(sample, tree.attribute)
        left = findleaf(sample, tree.left)
        right = findleaf(sample, tree.right)
        outcome = mpc.if_else(value, right.outcome_class, left.outcome_class)
        pruned = mpc.if_else(value, right.pruned, left.pruned)
        return Leaf(outcome, pruned)
    if isinstance(tree, Leaf):
        return tree


def getitem(sample, index):
    unit_vector = mpc.unit_vector(index, len(sample))
    return mpc.sum(mpc.schur_prod(sample, unit_vector))
