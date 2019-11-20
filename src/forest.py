from math import sqrt
from mpyc.runtime import mpc
from src.secint import secint
from src.dataset import ObliviousDataset, Sample
from src.train import train


def train_forest(samples, amount, depth, amount_of_features=None):
    if not amount_of_features:
        amount_of_features = int(sqrt(len(samples)))

    forest = []
    for _ in range(amount):
        selection = random_attributes(samples, amount_of_features)
        selection = bootstrap(selection)
        tree = train(selection, depth)
        forest.append(tree)
    return forest


def bootstrap(samples):
    return ObliviousDataset([samples.choice() for _ in range(len(samples))])


def random_attributes(samples, amount):
    columns = random_columns(samples, amount)
    outcomes = samples.outcomes
    smaller_samples = []
    for r in range(len(samples)):
        inputs = []
        outcome = outcomes[r]
        for c in range(len(columns)):
            inputs.append(columns[c][r])
        smaller_samples.append(Sample(inputs, outcome))
    return ObliviousDataset(smaller_samples)


def random_columns(samples, amount):
    indices = range(samples.number_of_attributes)
    selected = mpc.random.sample(secint, indices, amount)
    return [samples.column(index) for index in selected]
