from src.tree import Branch, Leaf
from mpyc.runtime import mpc


def classify(sample, tree):
    return findleaf(sample, tree).outcome_class


def findleaf(sample, tree):
    if isinstance(tree, Branch):
        value = getitem(sample, tree.attribute)
        left = findleaf(sample, tree.left)
        right = findleaf(sample, tree.right)
        take_left = mpc.if_else(value, right.pruned, (1 - left.pruned))
        outcome = mpc.if_else(
            take_left, left.outcome_class, right.outcome_class)
        pruned = mpc.and_(left.pruned, right.pruned)
        return Leaf(outcome, pruned)
    if isinstance(tree, Leaf):
        return tree


def getitem(sample, index):
    unit_vector = mpc.unit_vector(index, len(sample))
    return mpc.sum(mpc.schur_prod(sample, unit_vector))
