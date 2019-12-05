from mpyc.runtime import mpc

from src.if_else import if_else
from src.tree import Branch, Leaf


def classify(sample, tree):
    return findleaf(sample, tree).outcome


def findleaf(sample, tree):
    if isinstance(tree, Branch):
        value = get_attribute_value_for_sample(sample, tree.attribute)
        left = findleaf(sample, tree.left)
        right = findleaf(sample, tree.right)
        take_right = value > tree.threshold
        take_left = if_else(take_right, right.pruned, (1 - left.pruned))
        outcome = if_else(take_left, left.outcome, right.outcome)
        pruned = mpc.and_(left.pruned, right.pruned)
        return Leaf(outcome, pruned)
    if isinstance(tree, Leaf):
        return tree


def get_attribute_value_for_sample(sample, index):
    unit_vector = mpc.unit_vector(index, len(sample))
    return mpc.sum(mpc.schur_prod(sample, unit_vector))
