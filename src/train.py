from mpyc.runtime import mpc

from src.best_split import select_best_attribute
from src.partition import partition_binary
from src.secint import secint as s
from src.tree import Branch, Leaf


def train(samples, depth):
    print(f'Training at depth: {depth}')
    if depth > 0:
        best_attribute = select_best_attribute(samples)
        (samples_left, samples_right) = partition_binary(samples, best_attribute)
        left = train(samples_left, depth=depth-1)
        right = train(samples_right, depth=depth-1)
        return Branch(best_attribute, left=left, right=right)
    else:
        pruned = samples.len() == s(False)
        outcome = samples.determine_class()
        return Leaf(outcome, pruned)
