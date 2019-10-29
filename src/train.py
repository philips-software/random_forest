from src.tree import Node
from mpyc.runtime import mpc
from src.best_split import select_best_attribute

s = mpc.SecInt()


def train(samples):
    best_attribute = select_best_attribute(samples)
    return Node(best_attribute)
