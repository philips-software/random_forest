from src.tree import Branch, Leaf
from mpyc.runtime import mpc


def classify(sample, tree):
    if isinstance(tree, Branch):
        attribute_value = getitem(sample, tree.attribute)
        left_outcome = classify(sample, tree.left)
        right_outcome = classify(sample, tree.right)
        return mpc.if_else(attribute_value, right_outcome, left_outcome)
    if isinstance(tree, Leaf):
        return tree.outcome_class


def getitem(sample, index):
    unit_vector = mpc.unit_vector(index, len(sample))
    return mpc.sum(mpc.schur_prod(sample, unit_vector))
