from src.tree import Branch, Leaf
from mpyc.runtime import mpc


def classify(sample, tree):
    if isinstance(tree, Branch):
        attribute_index = tree.attribute
        unit_vector = mpc.unit_vector(attribute_index, len(sample))
        attribute_value = mpc.sum(mpc.schur_prod(sample, unit_vector))
        return mpc.if_else(
            attribute_value,
            classify(sample, tree.right),
            classify(sample, tree.left))
    if isinstance(tree, Leaf):
        return tree.outcome_class
