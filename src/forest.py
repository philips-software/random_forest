from src.dataset import ObliviousDataset
from src.train import train


def train_forest(samples, amount, depth):
    return [train(bootstrap(samples), depth) for _ in range(amount)]


def bootstrap(samples):
    return ObliviousDataset([samples.choice() for _ in range(len(samples))])
