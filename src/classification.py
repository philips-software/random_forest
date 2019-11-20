from src.tree import Branch, Leaf
from mpyc.runtime import mpc


def classify(sample, tree):
    if isinstance(tree, Branch):
        return mpc.if_else(
            getitem(sample, tree.attribute),
            classify(sample, tree.right),
            classify(sample, tree.left))
    if isinstance(tree, Leaf):
        return tree.outcome_class


def getitem(sample, index):
    unit_vector = mpc.unit_vector(index, len(sample))
    return mpc.sum(mpc.schur_prod(sample, unit_vector))
