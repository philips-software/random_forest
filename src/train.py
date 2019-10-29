from mpyc.runtime import mpc

from src.best_split import select_best_attribute
from src.partition import partition_binary
from src.secint import secint as s
from src.tree import Node


def train(samples, depth=1):
    best_attribute = select_best_attribute(samples)
    (samples_left, samples_right) = partition_binary(samples, best_attribute)
    left = train(samples_left, depth=depth-1) if depth > 1 else None
    right = train(samples_right, depth=depth-1) if depth > 1 else None
    return Node(best_attribute, left=left, right=right)
