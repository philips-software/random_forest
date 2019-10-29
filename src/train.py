from mpyc.runtime import mpc

from src.best_split import select_best_attribute
from src.secint import secint as s
from src.tree import Node


def train(samples):
    best_attribute = select_best_attribute(samples)
    return Node(best_attribute)
