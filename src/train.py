from mpyc.runtime import mpc

from src.best_split import select_best_attribute
from src.partition import partition_binary
from src.secint import secint as s
from src.tree import Branch


def train(samples, depth):
    best_attribute = select_best_attribute(samples)
    print(f'Training at depth: {depth}')
    (samples_left, samples_right) = partition_binary(samples, best_attribute)
    left = train(samples_left, depth=depth-1) if depth > 1 else None
    right = train(samples_right, depth=depth-1) if depth > 1 else None
    return Branch(best_attribute, left=left, right=right)
